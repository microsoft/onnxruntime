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
    int64_t H;
    int64_t W;
    int64_t num_input_channels;
    int64_t num_output_channels;
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
    const TensorShape& input_shape = X->Shape();

    // input validations
    if (group <= 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "group count is <= 0",
                             " group: ", group);
    }

    if (input_shape.NumDimensions() != 4) {
      // This condition is not true for two tests in ONNX tests series:
      // test_convtranspose_1d_cpu, test_convtranspose_3d_cpu.
      // TODO: the error message should tell which operator raises it.
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input X must be 4-dimensional.",
                             " X: ", X->Shape().ToString().c_str());
    }

    if (input_shape.NumDimensions() != F->Shape().NumDimensions()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "X num_dims does not match W num_dims.",
                             " X: ", X->Shape().ToString().c_str(),
                             " W: ", F->Shape().ToString().c_str());
    }

    const int64_t num_input_channels = input_shape[1];

    if (F->Shape()[0] != num_input_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "filter number not equal to input channel number.",
                             " filter_number: ", F->Shape()[0],
                             " num_input_channels: ", num_input_channels);
    }

    const int64_t N = input_shape[0];
    const int64_t H = input_shape[2];
    const int64_t W = input_shape[3];
    const int64_t num_output_channels_multiplier = F->Shape()[1];
    const int64_t num_output_channels = num_output_channels_multiplier * group;

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
    local_pads.reserve(2 * (input_shape.NumDimensions() - 2));
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
                              local_strides, local_dilations, local_output_padding, &local_pads, &Y_dims);
    TensorShape Yshape(Y_dims);
    Tensor* Y = context->Output(0, Yshape);

    p.X = X;
    p.F = F;
    p.B = B;
    p.Y = Y;
    p.N = N;
    p.H = H;
    p.W = W;
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
                                 const std::vector<int64_t>& p_dilations, const std::vector<int64_t>& p_output_padding,
                                 std::vector<int64_t>* p_pads, std::vector<int64_t>* output_shape_p) const {
    const int64_t N = input_shape[0];
    const int64_t H = input_shape[2];
    const int64_t W = input_shape[3];
    int64_t output_height = -1;
    int64_t output_width = -1;
    size_t output_shape_size = output_shape.size();

    if (output_shape_size != 0) {
      output_height = output_shape[output_shape_size - 2];
      output_width = output_shape[output_shape_size - 1];
      ORT_ENFORCE(output_height >= H, "Output height cannot be smaller than input height.");
      ORT_ENFORCE(output_width >= W, "Output width cannot be smaller than input width.");
    }

    ComputeTransposePadAndOutputShape(
        H,
        p_strides[0],
        kernel_shape[0],
        p_dilations[0],
        p_output_padding[0],
        auto_pad,
        &p_pads->at(0),
        &p_pads->at(2),
        &output_height);

    ComputeTransposePadAndOutputShape(
        W,
        p_strides[1],
        kernel_shape[1],
        p_dilations[1],
        p_output_padding[1],
        auto_pad,
        &p_pads->at(1),
        &p_pads->at(3),
        &output_width);

    output_shape_p->insert(output_shape_p->begin(), {N, output_channel, output_height, output_width});
  }

  const std::vector<int64_t> output_padding;
  const std::vector<int64_t> output_shape;

private:
  void ComputeTransposePadAndOutputShape (
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
      int64_t paddings = std::max<int64_t>(0, (in_size - 1) * stride + kernel + dilation - 1 + adj - *out_size);
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
          *out_size = (in_size - 1) * stride + kernel + dilation - 1 + adj;
          break;
        default:
          throw NotImplementedException("pad type not supported");
      }
    } else {
      *out_size =
          (in_size - 1) * stride + kernel + dilation - 1 + adj - *pad_head - *pad_tail;
    }
  }
};

}  // namespace onnxruntime
