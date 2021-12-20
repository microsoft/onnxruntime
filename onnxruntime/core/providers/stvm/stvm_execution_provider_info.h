// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef STVM_EXECUTION_PROVIDER_INFO_H
#define STVM_EXECUTION_PROVIDER_INFO_H

#include <unordered_map>
#include <vector>
#include <string>

#include "core/framework/provider_options.h"

namespace onnxruntime {

constexpr const char* default_target_str = "cpu";
constexpr const char* llvm_target_str = "llvm";

constexpr const char* cpu_target_str = "cpu";
constexpr const char* gpu_target_str = "gpu";

namespace stvm_cpu_targets {
// TODO(vvchernov): avx and avx512 need more careful differentiation for target
const std::string LLVM_TARGET_AVX = "llvm -mcpu=corei7-avx";
const std::string LLVM_TARGET_AVX2 = "llvm -mcpu=core-avx2";
const std::string LLVM_TARGET_SKYLAKE_AVX512 = "llvm -mcpu=skylake-avx512";
const std::string LLVM_TARGET_AVX512 = "llvm -mcpu=skylake-avx512";
}

constexpr const unsigned int default_opt_level = 3;

using STVMInputShapes = std::unordered_map<std::string, std::vector<int64_t>>;

// Information needed to construct an TVM execution provider.
struct StvmExecutionProviderInfo {
  std::string target{default_target_str};
  std::string target_host{default_target_str};
  unsigned int opt_level{default_opt_level};
  bool freeze_weights = true;
  bool to_nhwc = false;
  std::string tuning_file_path{""};
  std::string tuning_type{"AutoTVM"};
  std::string input_names_str{""};
  std::string input_shapes_str{""};
  STVMInputShapes input_shapes{};

  static std::string whitespace_trimming(const std::string& str);
  static StvmExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static StvmExecutionProviderInfo FromOptionsString(const char* options);
};

}  // namespace onnxruntime

#endif  // STVM_EXECUTION_PROVIDER_INFO_H
