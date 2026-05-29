/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/* Modifications Copyright (c) Microsoft. */
// Copyright (c) 2023 NVIDIA Corporation.

#pragma once

#include <algorithm>

#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/common/safeint.h"

namespace onnxruntime {

struct ConvTransposeAttributes : public ConvAttributes {
  template <typename KernelInfoType>
  explicit ConvTransposeAttributes(const KernelInfoType& info)
      : ConvAttributes(info),
        output_padding(info.GetAttrsOrDefault("output_padding")),
        output_shape(info.GetAttrsOrDefault("output_shape")) {
  }

  struct Prepare {
    const Tensor* X;
    const Tensor* F;
    const Tensor* B;
    Tensor* Y;
    int64_t N;
    int64_t num_input_channels;
    int64_t num_output_channels;
    TensorShape input_shape;
    TensorShapeVector kernel_shape;
    ConvPadVector pads;
    TensorShapeVector dilations;
    TensorShapeVector strides;
  };

  // Viewing dim 1 of the X input as 'input channels' (C) and dim 1 of the Y output as 'output channels' (M),
  // if is_nhwc is true, the input channels (dim 0) or output channels (dim 1) of the W input (the filter with
  // shape {C, M/group, ...}) could be transposed to be last. transposed_input_channels indicates whether dim 0 or
  // dim 1 was moved.
  //
  // e.g. XNNPACK moves the input channels dim to the end. CUDA moves the output channels dim to the end.
  Status PrepareForCompute(OpKernelContext* context, bool has_bias, Prepare& p,
                           bool dynamic_padding = false, const TensorShape* filter_shape = nullptr,
                           bool is_nhwc = false, bool transposed_input_channels = true) const {
    const Tensor* X = context->Input<Tensor>(0);
    const Tensor* F = (filter_shape != nullptr) ? nullptr : context->Input<Tensor>(1);
    const TensorShape& F_Shape = (filter_shape != nullptr) ? *filter_shape : F->Shape();
    const Tensor* Pads = dynamic_padding ? context->Input<Tensor>(2) : nullptr;
    const Tensor* B = has_bias ? (dynamic_padding ? context->Input<Tensor>(3) : context->Input<Tensor>(2)) : nullptr;

    const int rank = static_cast<int>(X->Shape().NumDimensions());

    // ConvTranspose requires X shape (N x C x D1...Dn) and W shape (C x M/group x k1...kn),
    // both must have at least 3 dimensions.
    if (rank < 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input X must have at least 3 dimensions (N x C x D1...Dn).",
                             " X: ", X->Shape().ToString().c_str());
    }

    if (static_cast<int>(F_Shape.NumDimensions()) < 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Filter W must have at least 3 dimensions (C x M/group x k1...kn).",
                             " W: ", F_Shape.ToString().c_str());
    }

    TensorShape input_shape = X->Shape().Slice(is_nhwc ? 1 : 2, is_nhwc ? rank - 1 : rank);
    const int64_t num_input_channels = is_nhwc ? X->Shape()[rank - 1] : X->Shape()[1];
    const int64_t N = X->Shape()[0];

    // W is {C, M/group, ....}. adjust for NHWC and transposed_input_channels
    // If we transposed the input channels, {C, M/group, ...} becomes {M/group, ..., C}
    // If we transposed the output channels, {C, M/group, ...} becomes {C, ..., M/group}
    const auto M_div_group_dim = is_nhwc ? (transposed_input_channels ? 0 : F_Shape.NumDimensions() - 1) : 1;
    const int64_t num_output_channels_multiplier = F_Shape[M_div_group_dim];
    const int64_t num_output_channels = num_output_channels_multiplier * group;

    // input validations
    if (group <= 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "group count is <= 0",
                             " group: ", group);
    }

    if (X->Shape().NumDimensions() != F_Shape.NumDimensions()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "X num_dims does not match W num_dims.",
                             " X: ", X->Shape().ToString().c_str(),
                             " W: ", F_Shape.ToString().c_str());
    }

    const auto F_channels_dim = is_nhwc && transposed_input_channels ? F_Shape.NumDimensions() - 1 : 0;
    if (F_Shape[F_channels_dim] != num_input_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "filter number not equal to input channel number.",
                             " filter_number: ", F_Shape[F_channels_dim],
                             " num_input_channels: ", num_input_channels);
    }

    // it looks like num_output_channels is really k*group similar to how in the conv case
    // num_input_channels is k*group. hence removing the check for num_output_channels here.

    if (num_input_channels % group != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input channels is not divisible by group.",
                             " num_input_channels: ", num_input_channels,
                             " group: ", group);
    }

    // Bias shape validation (It should be a 1D tensor with size M)
    // See https://github.com/microsoft/onnxruntime/issues/26144
    if (B != nullptr) {
      if (B->Shape().NumDimensions() != 1 || B->Shape()[0] != num_output_channels) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Bias shape is not compatible with number of output channels."
                               " It should be a 1-D tensor with size num_output_channels(M).",
                               " Bias: ", B->Shape(),
                               " num_output_channels: ", num_output_channels);
      }
    }

    TensorShapeVector kernel_shape;
    ORT_RETURN_IF_ERROR(ComputeKernelShape(F_Shape, kernel_shape, is_nhwc));

    TensorShapeVector local_output_padding(output_padding);
    if (local_output_padding.empty()) {
      local_output_padding.resize(kernel_shape.size(), 0);
    }
    if (local_output_padding.size() != kernel_shape.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "output_padding size (", local_output_padding.size(),
                             ") does not match the number of spatial dimensions (", kernel_shape.size(), ").");
    }
    ConvPadVector local_pads;
    local_pads.reserve(2 * (input_shape.NumDimensions()));
    if (dynamic_padding) {
      if (Pads->Shape().NumDimensions() != 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Dynamic pads tensor must be 1-D. Got rank: ", Pads->Shape().NumDimensions());
      }
      const int64_t expected_pads_size = SafeInt<int64_t>(kernel_shape.size()) * 2;
      const int64_t actual_pads_size = Pads->Shape().SizeFromDimension(0);
      if (actual_pads_size != expected_pads_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Dynamic pads tensor size (", actual_pads_size,
                               ") does not match expected size (2 * spatial_dims = ", expected_pads_size, ").");
      }
      const auto* pads_data = Pads->Data<int64_t>();
      for (int64_t i = 0; i < actual_pads_size; ++i) {
        if (pads_data[i] < 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Dynamic pads must be non-negative. Got pads[", i, "] = ", pads_data[i]);
        }
        local_pads.push_back(pads_data[i]);
      }
    } else {
      local_pads.assign(pads.begin(), pads.end());
    }
    if (local_pads.empty()) {
      local_pads.resize(kernel_shape.size() * 2, 0);
    }
    TensorShapeVector local_dilations(dilations);
    if (local_dilations.empty()) {
      local_dilations.resize(kernel_shape.size(), 1);
    }
    TensorShapeVector local_strides(strides);
    if (local_strides.empty()) {
      local_strides.resize(kernel_shape.size(), 1);
    }

    if (local_strides.size() != kernel_shape.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "strides size (", local_strides.size(),
                             ") does not match the number of spatial dimensions (", kernel_shape.size(), ").");
    }
    if (local_dilations.size() != kernel_shape.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "dilations size (", local_dilations.size(),
                             ") does not match the number of spatial dimensions (", kernel_shape.size(), ").");
    }

    // ONNX spec: "output_padding[i] should be less than max(stride[i], dilation[i])".
    // This constraint ensures the output_padding is unambiguous — larger values would shift
    // the output by more than one stride/dilation step, making the inverse of Conv ill-defined.
    for (size_t i = 0; i < local_output_padding.size(); ++i) {
      int64_t limit = std::max(local_strides[i], local_dilations[i]);
      if (local_output_padding[i] >= limit) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "output_padding[", i, "] (", local_output_padding[i],
                               ") must be less than max(stride, dilation) (", limit,
                               ") for spatial dimension ", i, ".");
      }
    }

    TensorShapeVector Y_dims;

    ORT_RETURN_IF_ERROR(ComputePadsAndOutputShape(input_shape, num_output_channels, kernel_shape,
                                                  local_strides, local_dilations, local_output_padding, N, &local_pads, &Y_dims, is_nhwc));
    TensorShape Yshape(Y_dims);
    Tensor* Y = context->Output(0, Yshape);

    p.X = X;
    p.F = F;
    p.B = B;
    p.Y = Y;
    p.N = N;
    p.input_shape = std::move(input_shape);
    p.num_input_channels = num_input_channels;
    p.num_output_channels = num_output_channels;
    p.kernel_shape = std::move(kernel_shape);
    p.pads = std::move(local_pads);
    p.strides = std::move(local_strides);
    p.dilations = std::move(local_dilations);
    return Status::OK();
  }

  Status ComputePadsAndOutputShape(TensorShape input_shape, int64_t output_channel,
                                   const TensorShapeVector& kernel_shape, const TensorShapeVector& p_strides,
                                   const TensorShapeVector& p_dilations, const TensorShapeVector& p_output_padding, const int64_t N,
                                   ConvPadVector* p_pads, TensorShapeVector* output_shape_p,
                                   bool is_nhwc = false) const {
    size_t output_shape_size = output_shape.size();
    size_t rank = input_shape.NumDimensions();

    if (p_pads->size() != 2 * rank) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "pads size (", p_pads->size(), ") does not match expected size (2 * ", rank, ").");
    }

    // output_shape attribute, if specified, must have either 'rank' or 'rank + 2' elements
    if (output_shape_size != 0 && output_shape_size != rank && output_shape_size != rank + 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "output_shape attribute has ", output_shape_size,
                             " elements, expected ", rank, " or ", rank + 2, ".");
    }

    if (is_nhwc) {
      output_shape_p->insert(output_shape_p->begin(), {N});
    } else {
      output_shape_p->insert(output_shape_p->begin(), {N, output_channel});
    }

    for (size_t dim = 0; dim < rank; ++dim) {
      int64_t dim_size = -1;

      if (output_shape_size != 0) {
        dim_size = output_shape_size == rank ? output_shape[dim] : output_shape[dim + 2];
      }

      ORT_RETURN_IF_ERROR(ComputeTransposePadAndOutputShape(
          input_shape[dim],
          p_strides[dim],
          kernel_shape[dim],
          p_dilations[dim],
          p_output_padding[dim],
          auto_pad,
          &(*p_pads)[dim],
          &(*p_pads)[input_shape.NumDimensions() + dim],
          &dim_size));

      if (dim_size <= 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Computed output dimension is <= 0 for dim ", dim,
                               ". Input shape: ", input_shape.ToString());
      }
      output_shape_p->push_back(dim_size);
    }
    if (is_nhwc) {
      output_shape_p->push_back(output_channel);
    }
    return Status::OK();
  }

  TensorShapeVector output_padding;
  TensorShapeVector output_shape;

 private:
  Status ComputeTransposePadAndOutputShape(
      const int64_t in_size,
      const int64_t stride,
      const int64_t kernel,
      const int64_t dilation,
      const int64_t adj,
      AutoPadType pad_type,
      int64_t* pad_head,
      int64_t* pad_tail,
      int64_t* out_size) const {
    // Output shape is explicitly provided - pad values will have to be computed
    if (*out_size != -1) {
      if (*out_size < 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Explicit output size is negative: ", *out_size);
      }
      // total pad
      auto total_pad = ComputeTotalPad(in_size, stride, adj,
                                       kernel, dilation, *out_size);
      DistributePadding(pad_type, total_pad, *pad_head, *pad_tail);
      return Status::OK();
    }

    // Output shape is not provided - it needs to be computed along with pad values (if applicable)

    // Validate that stride, kernel, and dilation are positive
    if (stride <= 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Stride must be positive. Got: ", stride);
    }
    if (kernel <= 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Kernel size must be positive. Got: ", kernel);
    }
    if (dilation <= 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Dilation must be positive. Got: ", dilation);
    }
    if (adj < 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Output padding must be non-negative. Got: ", adj);
    }

    // Compute padding if the auto_pad attribute is SAME_UPPER/SAME_LOWER
    if (pad_type == AutoPadType::SAME_UPPER || pad_type == AutoPadType::SAME_LOWER) {
      // The ONNX spec says if `auto_pad` attribute is set, pad until the `out_size`
      // is `in_size * stride`
      int64_t auto_out_size = SafeInt<int64_t>(in_size) * stride;
      auto total_pad = ComputeTotalPad(in_size, stride, adj,
                                       kernel, dilation, auto_out_size);
      DistributePadding(pad_type, total_pad, *pad_head, *pad_tail);
    }

    // *out_size = (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1 - *pad_head - *pad_tail
    *out_size = SafeInt<int64_t>(in_size - 1) * stride + adj +
                SafeInt<int64_t>(kernel - 1) * dilation + 1 -
                *pad_head - *pad_tail;
    return Status::OK();
  }
};

}  // namespace onnxruntime
