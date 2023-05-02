// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_EXECUTION_PROVIDER_OPTIONS_H
#define TVM_EXECUTION_PROVIDER_OPTIONS_H

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>

#include "core/framework/provider_options.h"
#include "core/framework/tensor_shape.h"

#include "tvm_defaults.h"

namespace onnxruntime {

namespace tvm {
namespace cpu_targets {
// TODO(vvchernov): avx and avx512 need more careful differentiation for target
const std::string LLVM_TARGET_AVX = "llvm -mcpu=corei7-avx";
const std::string LLVM_TARGET_AVX2 = "llvm -mcpu=core-avx2";
const std::string LLVM_TARGET_SKYLAKE_AVX512 = "llvm -mcpu=skylake-avx512";
const std::string LLVM_TARGET_AVX512 = "llvm -mcpu=skylake-avx512";
}  // namespace cpu_targets

using TVMTensorShapes = std::vector<TensorShapeVector>;
using TVMInputShapes = std::unordered_map<std::string, TensorShapeVector>;
using InputsInfoMap = std::unordered_map<size_t, TensorShapeVector>;

// Information needed to construct an TVM execution provider.
struct TvmEPOptions {
  std::string executor{tvm::default_executor_type};
  std::string so_folder{""};
  bool check_hash = false;
  std::string hash_file_path{""};
  std::string target{tvm::default_target_str};
  std::string target_host{tvm::default_target_str};
  unsigned int opt_level{tvm::default_opt_level};
  bool freeze_weights = true;
  bool to_nhwc = false;
  bool set_output_zero_copy = true;
  std::string tuning_file_path{""};
  std::string tuning_type{tvm::default_tuning_type};
  std::string input_names_str{""};
  std::string input_shapes_str{""};
  TVMInputShapes input_shapes{};
  TVMTensorShapes output_shapes{};
};

std::ostream& operator<<(std::ostream& out, const TvmEPOptions& options);

class TvmEPOptionsHelper {
 public:
  static TvmEPOptions FromOptionsString(const char* options);
  static TvmEPOptions FromProviderOptions(const ProviderOptions& options);
  static std::string whitespace_trimming(const std::string& str);

  static bool checkCPUTarget(const std::string& target);
  static bool checkGPUTarget(const std::string& target);

 private:
  static void optionsPostprocess(TvmEPOptions& options);
  static void setInputShapes(TvmEPOptions& options);
  static void targetPostprocess(std::string& target);
  static void ProcessCPUTarget(std::string& target);
  static void ProcessGPUTarget();
  static void targetHostPostprocess(const std::string& target, std::string& target_host);
  static void optLevelPostprocess(unsigned int& opt_level);
};

}  // namespace tvm
}  // namespace onnxruntime

#endif  // TVM_EXECUTION_PROVIDER_OPTIONS_H
