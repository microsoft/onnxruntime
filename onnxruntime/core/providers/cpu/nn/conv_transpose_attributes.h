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

#pragma once

#include "core/providers/cpu/nn/conv_attributes.h"

namespace onnxruntime {

struct ConvTransposeAttributes : public ConvAttributes {
  explicit ConvTransposeAttributes(const OpNodeProtoHelper<ProtoHelperNodeContext>& info)
      : ConvAttributes(info),
        output_padding(info.GetAttrsOrDefault<int64_t>("output_padding")),
        output_shape(info.GetAttrsOrDefault<int64_t>("output_shape")) {
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
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    std::vector<int64_t> strides;
  };

  Status PrepareForCompute(OpKernelContext* context, bool has_bias, Prepare& p, bool dynamic_padding = false) const {
    const Tensor* X = context->Input<Tensor>(0);
    const Tensor* F = context->Input<Tensor>(1);
    const Tensor* Pads = dynamic_padding ? context->Input<Tensor>(2) : nullptr;
    const Tensor* B = has_bias ? (dynamic_padding ? context->Input<Tensor>(3) : context->Input<Tensor>(2)) : nullptr;
    const TensorShape& input_shape = X->Shape().Slice(2);

    const int64_t num_input_channels = X->Shape()[1];
    const int64_t N = X->Shape()[0];
    const int64_t num_output_channels_multiplier = F->Shape()[1];
    const int64_t num_output_channels = num_output_channels_multiplier * group;

    // input validations
    if (group <= 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "group count is <= 0",
                             " group: ", group);
    }

    if (X->Shape().NumDimensions() != F->Shape().NumDimensions()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "X num_dims does not match W num_dims.",
                             " X: ", X->Shape().ToString().c_str(),
                             " W: ", F->Shape().ToString().c_str());
    }

    if (F->Shape()[0] != num_input_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "filter number not equal to input channel number.",
                             " filter_number: ", F->Shape()[0],
                             " num_input_channels: ", num_input_channels);
    }

    // it looks like num_output_channels is really k*group similar to how in the conv case
    // num_input_channels is k*group. hence removing the check for num_output_channels here.

    if (num_input_channels % group != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input channels is not divisible by group.",
                             " num_input_channels: ", num_input_channels,
                             " group: ", group);
    }

    std::vector<int64_t> kernel_shape;
    ORT_RETURN_IF_ERROR(ComputeKernelShape(F->Shape(), kernel_shape));

    std::vector<int64_t> local_output_padding(output_padding);
    if (local_output_padding.empty()) {
      local_output_padding.resize(kernel_shape.size(), 0);
    }
    std::vector<int64_t> local_pads;
    local_pads.reserve(2 * (input_shape.NumDimensions()));
    if (dynamic_padding) {
      for (int64_t i = 0; i < Pads->Shape().SizeFromDimension(0); ++i) {
        local_pads.push_back(Pads->Data<int64_t>()[i]);
      }
    } else {
      local_pads.assign(pads.begin(), pads.end());
    }
    if (local_pads.empty()) {
      local_pads.resize(kernel_shape.size() * 2, 0);
    }
    std::vector<int64_t> local_dilations(dilations);
    if (local_dilations.empty()) {
      local_dilations.resize(kernel_shape.size(), 1);
    }
    std::vector<int64_t> local_strides(strides);
    if (local_strides.empty()) {
      local_strides.resize(kernel_shape.size(), 1);
    }

    std::vector<int64_t> Y_dims;

    ComputePadsAndOutputShape(input_shape, num_output_channels, kernel_shape,
                              local_strides, local_dilations, local_output_padding, N, &local_pads, &Y_dims);
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

  void ComputePadsAndOutputShape(TensorShape input_shape, int64_t output_channel,
                                 const std::vector<int64_t>& kernel_shape, const std::vector<int64_t>& p_strides,
                                 const std::vector<int64_t>& p_dilations, const std::vector<int64_t>& p_output_padding, const int64_t N,
                                 std::vector<int64_t>* p_pads, std::vector<int64_t>* output_shape_p) const {
    size_t output_shape_size = output_shape.size();
    output_shape_p->insert(output_shape_p->begin(), {N, output_channel});

    size_t rank = input_shape.NumDimensions();
    for (size_t dim = 0; dim < rank; ++dim) {
      int64_t dim_size = -1;

      if (output_shape_size != 0) {
        dim_size = output_shape_size == rank ? output_shape[dim] : output_shape[dim + 2];
      }

      ComputeTransposePadAndOutputShape(
          input_shape[dim],
          p_strides[dim],
          kernel_shape[dim],
          p_dilations[dim],
          p_output_padding[dim],
          auto_pad,
          &p_pads->at(dim),
          &p_pads->at(input_shape.NumDimensions() + dim),
          &dim_size);

      ORT_ENFORCE(dim_size > 0, "Invalid input shape: ", input_shape.ToString());
      output_shape_p->push_back(dim_size);
    }
  }

  const std::vector<int64_t> output_padding;
  const std::vector<int64_t> output_shape;

 private:
  void ComputeTransposePadAndOutputShape(
      const int64_t in_size,
      const int64_t stride,
      const int64_t kernel,
      const int64_t dilation,
      const int64_t adj,
      AutoPadType pad_type,
      int64_t* pad_head,
      int64_t* pad_tail,
      int64_t* out_size) const {
    if (*out_size != -1) {
      ORT_ENFORCE(*out_size >= 0);
      // total padding size
      int64_t paddings = std::max<int64_t>(0, (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1 - *out_size);
      if (pad_type == AutoPadType::SAME_UPPER) {  // pad more on head when paddings are odd.
        *pad_head = paddings - paddings / 2;
        *pad_tail = paddings / 2;
      } else {
        // for pad_type is NOTSET, SAME_LOWER or VALID
        // set pad_head as paddings/2, pad_tail as paddings-paddings/2.
        // That said, we pad more on tail when paddings are odd.
        *pad_head = paddings / 2;
        *pad_tail = paddings - paddings / 2;
      }
      return;
    }
    if (pad_type != AutoPadType::NOTSET) {
      switch (pad_type) {
          // We handle cases of AutoPadType::VALID and AutoPadType::SAME_UPPER/LOWER,
          // the same way
        case AutoPadType::VALID:
        case AutoPadType::SAME_UPPER:
        case AutoPadType::SAME_LOWER:
          *pad_head = 0;
          *pad_tail = 0;
          *out_size = (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1;
          break;
        default:
          ORT_NOT_IMPLEMENTED("pad type not supported");
      }
    } else {
      *out_size =
          (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1 - *pad_head - *pad_tail;
    }
  }
};

}  // namespace onnxruntime
