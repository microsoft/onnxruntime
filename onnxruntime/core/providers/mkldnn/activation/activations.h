/* Copyright(C) 2018 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions
and limitations under the License.
==============================================================================*/

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/activation/activations.h"

namespace onnxruntime {
namespace mkl_dnn {

template <typename T>
class Relu : public onnxruntime::Relu<T> {
 public:
  Relu(const OpKernelInfo& info) : onnxruntime::Relu<T>(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace mkl_dnn
}  // namespace onnxruntime