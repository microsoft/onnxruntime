// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_CORE_PROVIDERS_TVM_TVM_DEFAULTS_H_
#define ONNXRUNTIME_CORE_PROVIDERS_TVM_TVM_DEFAULTS_H_

#include <string>

namespace onnxruntime {
namespace tvm {

namespace env_vars {
static const std::string kDumpSubgraphs = "ORT_TVM_DUMP_SUBGRAPHS";
}  // namespace env_vars

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

#endif  // ONNXRUNTIME_CORE_PROVIDERS_TVM_TVM_DEFAULTS_H_
