// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_DEFAULTS_H
#define TVM_DEFAULTS_H

namespace onnxruntime {
namespace tvm {

constexpr const char* default_executor_type = "vm";
constexpr const char* vm_executor_type = "vm";
constexpr const char* graph_executor_type = "graph";

constexpr const char* default_target_str = "llvm";
constexpr const char* llvm_target_str = "llvm";

constexpr const char* cpu_target_str = "cpu";
constexpr const char* gpu_target_str = "gpu";

constexpr const char* default_tuning_type = "AutoTVM";
constexpr const char* autotvm_tuning_type = "AutoTVM";
constexpr const char* ansor_tuning_type = "Ansor";

constexpr const unsigned int default_opt_level = 3;

}  // namespace tvm
}  // namespace onnxruntime

#endif // TVM_DEFAULTS_H