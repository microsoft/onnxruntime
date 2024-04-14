// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <dlpack/dlpack.h>
#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {
namespace aten_ops {

typedef bool (*IsCpuArgumentFunc)(const char* op_name, const char* overload_name, size_t index, bool is_input);
typedef void (*ExecuteATenOperatorFunc)(const char* op_name, const char* overload_name, size_t input_size,
                                        DLManagedTensor** dlpack_inputs, size_t output_size,
                                        DLManagedTensor** dlpack_outputs);

class ATenOperatorExecutor {
 public:
  static ATenOperatorExecutor& Instance() {
    static ATenOperatorExecutor instance;
    return instance;
  }

  void Initialize(void* p_is_cpu_argument_func_raw, void* p_execute_aten_op_func_raw) {
    ORT_ENFORCE(p_is_cpu_argument_func_raw && p_execute_aten_op_func_raw);
    p_is_cpu_argument_func_ = reinterpret_cast<IsCpuArgumentFunc>(p_is_cpu_argument_func_raw);
    p_execute_aten_op_func_ = reinterpret_cast<ExecuteATenOperatorFunc>(p_execute_aten_op_func_raw);
  }

  bool IsInitialized() { return p_execute_aten_op_func_ != nullptr; }

  bool IsCpuArgument(const std::string& op_name, const std::string& overload_name, size_t index, bool is_input) {
    ORT_ENFORCE(p_is_cpu_argument_func_, "ATenOperatorExecutor is not initialized.");
    return p_is_cpu_argument_func_(op_name.c_str(), overload_name.c_str(), index, is_input);
  }

  void operator()(const std::string& op_name, const std::string& overload_name, size_t input_size,
                  DLManagedTensor** dlpack_inputs, size_t output_size, DLManagedTensor** dlpack_outputs) {
    ORT_ENFORCE(p_execute_aten_op_func_, "ATenOperatorExecutor is not initialized.");
    p_execute_aten_op_func_(op_name.c_str(), overload_name.c_str(), input_size, dlpack_inputs, output_size,
                            dlpack_outputs);
  }

 private:
  IsCpuArgumentFunc p_is_cpu_argument_func_ = nullptr;
  ExecuteATenOperatorFunc p_execute_aten_op_func_ = nullptr;
};

}  // namespace aten_ops
}  // namespace contrib
}  // namespace onnxruntime
