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

template <typename T>
class Conv : public OpKernel, public ConvBase {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info), ConvBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime

namespace onnxruntime {
namespace contrib {

template <typename T>
class ConvGrad final : public OpKernel, public ConvBase {
 public:
  explicit ConvGrad(const OpKernelInfo& info) : OpKernel(info), ConvBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConvGrad);
};

}  // namespace contrib
}  // namespace onnxruntime
