// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <dlpack/dlpack.h>
#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {
namespace aten_ops {

typedef std::vector<DLManagedTensor*> (*ExecuteATenOperatorFunc)(
    const char* op_name, const std::vector<std::tuple<size_t, DLManagedTensor*>>& tensor_arguments,
    const std::vector<std::tuple<size_t, int64_t>>& int_arguments,
    const std::vector<std::tuple<size_t, float>>& float_arguments,
    const std::vector<std::tuple<size_t, bool>>& bool_arguments);

class ATenOperatorExecutor {
 public:
  static ATenOperatorExecutor& Instance() {
    static ATenOperatorExecutor instance;
    return instance;
  }

  std::vector<DLManagedTensor*> operator()(const std::string& op_name,
                                           const std::vector<std::tuple<size_t, DLManagedTensor*>>& tensor_arguments,
                                           const std::vector<std::tuple<size_t, int64_t>>& int_arguments,
                                           const std::vector<std::tuple<size_t, float>>& float_arguments,
                                           const std::vector<std::tuple<size_t, bool>>& bool_arguments) {
    ORT_ENFORCE(p_func_);
    return p_func_(op_name.c_str(), tensor_arguments, int_arguments, float_arguments, bool_arguments);
  }

  void SetExecutorFunc(void* p_func_raw) { p_func_ = reinterpret_cast<ExecuteATenOperatorFunc>(p_func_raw); }

 private:
  ATenOperatorExecutor() = default;
  ExecuteATenOperatorFunc p_func_;
};

}  // namespace aten_ops
}  // namespace contrib
}  // namespace onnxruntime
