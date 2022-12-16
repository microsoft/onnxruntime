// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

struct DLManagedTensor;

namespace onnxruntime {
namespace contrib {
namespace torch {

typedef void (*ExecuteTorchScriptFunc)(const int64_t key, const char* script, size_t input_size,
                                       DLManagedTensor** dlpack_inputs, size_t output_size,
                                       DLManagedTensor** dlpack_outputs);

class TorchScriptExecutor {
 public:
  static TorchScriptExecutor& Instance() {
    static TorchScriptExecutor instance;
    return instance;
  }

  void Initialize(void* p_execute_torch_script_func_raw) {
    ORT_ENFORCE(p_execute_torch_script_func_raw);
    p_execute_torch_script_func_ = reinterpret_cast<ExecuteTorchScriptFunc>(p_execute_torch_script_func_raw);
  }

  bool IsInitialized() { return p_execute_torch_script_func_ != nullptr; }

  void operator()(const int64_t key, const std::string& script, size_t input_size, DLManagedTensor** dlpack_inputs,
                  size_t output_size, DLManagedTensor** dlpack_outputs) {
    ORT_ENFORCE(p_execute_torch_script_func_, "TorchScriptExecutor is not initialized.");
    p_execute_torch_script_func_(key, script.c_str(), input_size, dlpack_inputs, output_size, dlpack_outputs);
  }

 private:
  ExecuteTorchScriptFunc p_execute_torch_script_func_ = nullptr;
};

}  // namespace torch
}  // namespace contrib
}  // namespace onnxruntime
