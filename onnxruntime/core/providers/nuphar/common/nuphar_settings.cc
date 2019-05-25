// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/common/nuphar_settings.h"

#include "core/codegen/common/common.h"
#include "core/codegen/common/utils.h"
#include "core/common/logging/logging.h"
#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace onnxruntime {
namespace nuphar_codegen {

static const std::unordered_set<std::string> valid_keys = {
    codegen::CodeGenSettings::kDumpAllOptions,
    codegen::CodeGenSettings::kCodeGenDumpModule,
    codegen::CodeGenSettings::kCodeGenDumpLower,
    codegen::CodeGenSettings::kCodeGenDumpSchedule,
    codegen::CodeGenSettings::kCodeGenFastMath,
    kNupharDumpFusedNodes,
    kNupharMatmulExec,
    kNupharCachePath,
    kNupharCacheVersion,
    kNupharCacheSoName,
    kNupharCacheModelChecksum,
    kNupharCacheForceNoJIT,
    kNupharCodeGenTarget};

void CreateNupharCodeGenSettings() {
  std::map<std::string, std::string> options;

#ifndef GOLDEN_BUILD
  // environment variables override existing settings
  for (const auto& key : valid_keys) {
    std::string env_key;
    // env var is always upper case
    std::transform(key.begin(), key.end(), std::back_inserter(env_key), (int (*)(int))std::toupper);
    if (IsEnvVarDefined(env_key.c_str())) {
      // value is always lower case
      auto value = std::string(GetEnv(env_key.c_str()).get());

      if (options.count(key) > 0 && options.at(key) != value) {
        LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "NupharCodeGenSettings: option" << key << " is overridded by environment variable "
                                                 << env_key << " from: " << options.at(key) << " to: " << value;
      }

      std::string value_lower;
      std::transform(value.begin(), value.end(), std::back_inserter(value_lower), (int (*)(int))std::tolower);
      options[key] = value_lower;
    }
  }
#endif

  codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();

  // create two temporary strings to get rid of the odr-use issue introduced
  // The issue would trigger missing definition errors for static constexpr members
  // at link time.
  std::string fast_math_opt(codegen::CodeGenSettings::kCodeGenFastMath);
  std::string select_fast_math(codegen::CodeGenSettings::kCodeGenFastMath_ShortPolynormial);
  // set jit cache so name
  std::string cache_so_name_opt(nuphar_codegen::kNupharCacheSoName);
  std::string cache_so_name_default(nuphar_codegen::kNupharCacheSoName_default);

  settings.InsertOptions({{fast_math_opt, select_fast_math},
                          {cache_so_name_opt, cache_so_name_default}});

  settings.InsertOptions(options);

  if (settings.HasOption(codegen::CodeGenSettings::kDumpAllOptions)) {
    settings.DumpOptions();
  }
}

}  // namespace nuphar_codegen
}  // namespace onnxruntime
