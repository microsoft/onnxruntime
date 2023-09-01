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

#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <typename T>
class LpNorm final : public OpKernel {
 public:
  LpNorm(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_).IsOK());
    ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("p", &p_).IsOK());
    ORT_ENFORCE(p_ == 1 || p_ == 2);
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t axis_;
  int64_t p_;
};
}  // namespace onnxruntime
