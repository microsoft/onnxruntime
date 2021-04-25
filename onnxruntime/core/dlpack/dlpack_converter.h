// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/ml_value.h"
#include <dlpack/dlpack.h>

#ifdef USE_TORCH
#include <torch/torch.h>
#endif

// This convertor will take an OrtValue and wrap it as a DLPack tensor

namespace onnxruntime {
namespace dlpack {

DLManagedTensor* OrtValueToDlpack(OrtValue& ort_value);

// DLPack uses same config for both bool and unit8. Parameter is_bool_tensor is to
// tell ORT the data type when creating OrtValue.
OrtValue DlpackToOrtValue(DLManagedTensor* dlpack, bool is_bool_tensor = false);

#ifdef USE_TORCH
at::Tensor ToTorchTensor(OrtValue& ort_value);
OrtValue FromTorchTensor(const at::Tensor& torch_tensor);

template <class Result, class... Args>
Result GetATenOpAndExecute(const std::string& op_name, Args&&... args) {
  auto& ops = torch::jit::getAllOperatorsFor(torch::jit::Symbol::fromQualString(op_name));
  std::cout << "Op list size: " << ops.size() << std::endl;
  TORCH_INTERNAL_ASSERT(ops.size() == 1);

  auto& op = ops.front();
  std::cout << "Op name: " << op->schema().name() << std::endl;

  torch::jit::Stack stack;
  torch::jit::push(stack, std::forward<Args>(args)...);
  op->getOperation()(&stack);

  TORCH_INTERNAL_ASSERT(1 == stack.size());
  return torch::jit::pop(stack).to<Result>();
}

#endif

}  // namespace dlpack
}  // namespace onnxruntime
