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

#include "core/providers/cpu/nn/conv_base.h"

namespace onnxruntime {

class ConvTransposeBase : public ConvBase {
 protected:
  ConvTransposeBase(const OpKernelInfo& info)
      : ConvBase(info),
        output_padding_(info.GetAttrsOrDefault<int64_t>("output_padding")),
        output_shape_(info.GetAttrsOrDefault<int64_t>("output_shape")) {
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

  Status PrepareForCompute(OpKernelContext* context, bool has_bias, Prepare& p) const;

  void ComputePadsAndOutputShape(TensorShape input_shape, int64_t output_channel,
                                 const std::vector<int64_t>& kernel_shape, const std::vector<int64_t>& strides,
                                 const std::vector<int64_t>& dilations, const std::vector<int64_t>& output_padding,
                                 std::vector<int64_t>* pads, std::vector<int64_t>* output_shape) const;

  const std::vector<int64_t> output_padding_;
  const std::vector<int64_t> output_shape_;
};

template <typename T>
class ConvTranspose : public OpKernel, public ConvTransposeBase {
 public:
  ConvTranspose(const OpKernelInfo& info) : OpKernel(info), ConvTransposeBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
