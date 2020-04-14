// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/common/nuphar_settings.h"

#include "core/codegen/common/common.h"
#include "core/codegen/common/utils.h"
#include "core/common/logging/logging.h"
#include "core/providers/nuphar/nuphar_execution_provider.h"

#include <algorithm>
#include <cctype>
#include <unordered_set>
#include <regex>

namespace onnxruntime {
namespace nuphar {

static const std::unordered_set<std::string> valid_keys = {
    codegen::CodeGenSettings::kDumpAllOptions,
    codegen::CodeGenSettings::kCodeGenDumpModule,
    codegen::CodeGenSettings::kCodeGenDumpLower,
    codegen::CodeGenSettings::kCodeGenDumpSchedule,
    kNupharFastMath,
    kNupharFastActivation,
    kNupharForceNoTensorize,
    kNupharTensorize_IGEMM_Tile_M,
    kNupharTensorize_IGEMM_Tile_N,
    kNupharTensorize_IGEMM_Tile_K,
    kNupharTensorize_IGEMM_Permute,
    kNupharTensorize_IGEMM_Split_Last_Tile,
    kNupharDumpFusedNodes,
    kNupharDumpPartition,
    kNupharIMatMulForceMkl,
    kNupharMatmulExec,
    kNupharCachePath,
    kNupharCacheSoName,
    kNupharCacheModelChecksum,
    kNupharCacheForceNoJIT,
    kNupharCodeGenTarget,
    kNupharParallelMinWorkloads};

void SetDefaultOptions(std::map<std::string, std::string>& options) {
  // create two temporary strings to get rid of the odr-use issue introduced
  // The issue would trigger missing definition errors for static constexpr members
  // at link time.
  std::string fast_math_opt(kNupharFastMath);
  std::string select_fast_math(kNupharFastMath_ShortPolynormial);
  options.insert(std::make_pair(fast_math_opt, select_fast_math));

  std::string fast_act_opt(kNupharFastActivation);
  std::string select_fast_act(kNupharActivations_DeepCpu);
  options.insert(std::make_pair(fast_act_opt, select_fast_act));

  // set jit cache so name
  std::string cache_so_name_opt(kNupharCacheSoName);
  std::string cache_so_name_default(kNupharCacheSoName_Default);
  options.insert(std::make_pair(cache_so_name_opt, cache_so_name_default));

  std::string parallel_min_workloads_opt(kNupharParallelMinWorkloads);
#if defined(_OPENMP) || defined(USE_MKLML)
  // a rough estimate of workloads based on static dimensions for each thread, when using parallel schedule
  // user may change it to 0 to turn it off,
  // or use OMP_NUM_THREADS to control TVM thread pool similar to control MKL
  unsigned int parallel_min_workloads_default = 64;
#else
  // turn off parallel schedule by default to avoid TVM thread pool confliction with others
  // this is to ensure performance when user runs multiple inference threads, with each runs as single thread
  // if needed, user can override it with settings, and use TVM_NUM_THREADS to control the thread pool
  unsigned int parallel_min_workloads_default = 0;
#endif
  options.insert(std::make_pair(parallel_min_workloads_opt, std::to_string(parallel_min_workloads_default)));
}

void CreateNupharCodeGenSettings(const NupharExecutionProviderInfo& info) {
  std::map<std::string, std::string> options;
  SetDefaultOptions(options);

  std::unordered_set<std::string> required_options;
  if (!info.settings.empty()) {
    const std::string& str = info.settings;

    // tokenize settings
    std::regex reg("\\s*,\\s*");
    std::sregex_token_iterator iter(str.begin(), str.end(), reg, -1);
    std::sregex_token_iterator iter_end;
    std::vector<std::string> pairs(iter, iter_end);

    ORT_ENFORCE(pairs.size() > 0);
    for (const auto& pair : pairs) {
      auto pos_colon = pair.find(':');
      ORT_ENFORCE(pos_colon != std::string::npos, "Invalid key value pair.");
      std::string key = pair.substr(0, pos_colon);
      std::string value = pair.substr(pos_colon + 1);

      // trim leading and trailing spaces from key/value
      auto trim = [](const std::string& str) -> std::string {
        const std::string WHITESPACE = " \n\r\t\f\v";
        size_t start = str.find_first_not_of(WHITESPACE);
        if (start == std::string::npos) {
          return "";
        } else {
          size_t end = str.find_last_not_of(WHITESPACE);
          ORT_ENFORCE(end != std::string::npos);
          return str.substr(start, end + 1);
        }
      };
      key = trim(key);
      value = trim(value);

      if (valid_keys.count(key) == 0) {
        ORT_NOT_IMPLEMENTED("NupharCodeGenSettings: unknown option (", key, ")");
      }
      required_options.insert(key);
      options[key] = value;
    }
  }

#ifndef GOLDEN_BUILD
  // environment variables override existing settings
  for (const auto& key : valid_keys) {
    std::string env_key;
    // env var is always upper case
    std::transform(key.begin(), key.end(), std::back_inserter(env_key), (int (*)(int))std::toupper);
    if (IsEnvVarDefined(env_key.c_str())) {
      // value is case-sensitive
      auto value = std::string(GetEnv(env_key.c_str()).get());

      if (required_options.count(key) > 0 && options.at(key) != value) {
        LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL)
            << "NupharCodeGenSettings: option(" << key
            << ") from environment variable is ignored because of existing required option value: "
            << options.at(key);
      } else {
        options[key] = value;
      }
    }
  }
#endif

  codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
  settings.Clear();  // remove previous settings and start from scratch

  settings.InsertOptions(options);

  if (settings.HasOption(codegen::CodeGenSettings::kDumpAllOptions)) {
    settings.DumpOptions();
  }
}

}  // namespace nuphar
}  // namespace onnxruntime
