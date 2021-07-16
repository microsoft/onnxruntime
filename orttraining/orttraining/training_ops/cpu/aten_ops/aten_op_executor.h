// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <dlpack/dlpack.h>
#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {
namespace aten_ops {

typedef std::vector<DLManagedTensor*> (*ExecuteATenOperatorFunc)(
    const char* op_name, const std::vector<std::pair<size_t, DLManagedTensor*>>& tensor_arguments,
    const std::vector<std::pair<size_t, int64_t>>& int_arguments,
    const std::vector<std::pair<size_t, float>>& float_arguments,
    const std::vector<std::pair<size_t, bool>>& bool_arguments,
    const std::vector<std::pair<size_t, std::vector<int64_t>>>& int_array_arguments,
    const std::vector<std::pair<size_t, std::vector<float>>>& float_array_arguments,
    const std::vector<std::pair<size_t, std::vector<bool>>>& bool_array_arguments);

class ATenOperatorExecutor {
 public:
  static ATenOperatorExecutor& Instance() { return InstanceImpl(); }

  static void Initialize(void* p_func_raw) { InstanceImpl(p_func_raw); }

  std::vector<DLManagedTensor*> operator()(
      const std::string& op_name, const std::vector<std::pair<size_t, DLManagedTensor*>>& tensor_arguments,
      const std::vector<std::pair<size_t, int64_t>>& int_arguments,
      const std::vector<std::pair<size_t, float>>& float_arguments,
      const std::vector<std::pair<size_t, bool>>& bool_arguments,
      const std::vector<std::pair<size_t, std::vector<int64_t>>>& int_array_arguments,
      const std::vector<std::pair<size_t, std::vector<float>>>& float_array_arguments,
      const std::vector<std::pair<size_t, std::vector<bool>>>& bool_array_arguments) {
    ORT_ENFORCE(p_func_, "ATenOperatorExecutor is not initialized.");
    return p_func_(op_name.c_str(), tensor_arguments, int_arguments, float_arguments, bool_arguments,
                   int_array_arguments, float_array_arguments, bool_array_arguments);
  }

 private:
  static ATenOperatorExecutor& InstanceImpl(void* p_func_raw = nullptr) {
    static ATenOperatorExecutor instance(p_func_raw);
    return instance;
  }

  ATenOperatorExecutor(void* p_func_raw) {
    ORT_ENFORCE(p_func_raw);
    p_func_ = reinterpret_cast<ExecuteATenOperatorFunc>(p_func_raw);
  }

  ExecuteATenOperatorFunc p_func_;
};

}  // namespace aten_ops
}  // namespace contrib
}  // namespace onnxruntime
