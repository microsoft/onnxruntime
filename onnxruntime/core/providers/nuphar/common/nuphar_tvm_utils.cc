// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/common/nuphar_tvm_utils.h"

#include "core/providers/nuphar/common/nuphar_subgraph.h"
#include "core/providers/nuphar/common/nuphar_settings.h"
#include "core/codegen/common/common.h"
#include "core/codegen/common/target_info.h"

#include "core/common/logging/logging.h"
#include "core/platform/env.h"
#include "core/providers/common.h"
#include "gsl/gsl"
#include <topi/detail/extern.h>
#include <tvm/ir_pass.h>
#include <experimental/filesystem>
#include <fstream>
namespace fs = std::experimental::filesystem;

namespace onnxruntime {
namespace nuphar {

static bool GetOrCreateTVMModuleCacheDirectory(fs::path& path, bool create) {
  codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();

  if (!settings.HasOption(kNupharCachePath))
    return false;

  std::string version;
  if (settings.HasOption(kNupharCacheVersion)) {
    version = settings.GetOptionValue(kNupharCacheVersion);
  } else {
    version = kNupharCacheVersion_Current;
  }

  path = settings.GetOptionValue(kNupharCachePath);
  if (!create && !fs::is_directory(path))
    return false;

  if (!fs::is_directory(path))
    if (!fs::create_directory(path)) {
      throw std::runtime_error("Failed to create directory " + path.string());
    }

  path.append(version);
  if (!create && !fs::is_directory(path))
    return false;

  if (!fs::is_directory(path))
    if (!fs::create_directory(path)) {
      throw std::runtime_error("Failed to create directory " + path.string());
    }

  return true;
}

static bool GetCacheSoFilePath(std::string& so_path) {
  codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
  fs::path path;
  if (!GetOrCreateTVMModuleCacheDirectory(path, /*create*/ false))
    return false;

  auto so_name = settings.GetOptionValue(kNupharCacheSoName);
  path.append(so_name);
  if (fs::is_regular_file(path)) {
    so_path = path.string();
    return true;
  }
  return false;
}

static void* GetFuncFromLibrary(const std::string& so_path, const std::string& func_name, bool throw_if_not_found = true) {
  void* so_handle;
  ORT_ENFORCE(Env::Default().LoadDynamicLibrary(so_path, &so_handle).IsOK());
  void* func = nullptr;
  Status s = Env::Default().GetSymbolFromLibrary(so_handle, func_name, &func);
  if (throw_if_not_found && !s.IsOK())
    ORT_ENFORCE(false, "Cannot find ", func_name, " in ", so_path);
  return func;
}

static bool disable_caching_due_to_checksum_failure = false;

static bool VerifyTVMModuleChecksum(const std::string& so_path) {
  static std::string last_so_path;
  static bool last_checksum_validated = false;
  static std::mutex checksum_mutex;
  if (last_so_path != so_path) {
    std::lock_guard<std::mutex> lock(checksum_mutex);
    if (last_so_path != so_path) {
      disable_caching_due_to_checksum_failure = false;  // reset disabled caching for a new file
      last_so_path = so_path;
      void* f = GetFuncFromLibrary(so_path, "_ORTInternal_GetCheckSum", /*throw_if_not_found*/ false);
      if (f) {
        typedef void (*GetChecksumFunc)(const char*&, size_t&);
        GetChecksumFunc func = reinterpret_cast<GetChecksumFunc>(f);
        const char* model_checksum;
        size_t model_checksum_len;
        func(model_checksum,
             model_checksum_len);

        codegen::CodeGenSettings& setting = codegen::CodeGenSettings::Instance();
        // When checksum is expected by dll/so, user must set environment variable
        // NUPHAR_CACHE_MODEL_CHECKSUM from md5 digest of running model.
        // User may choose to run with base model or simplified mode and any match
        // would be regarded as validated.
        // Note that checksum validation here is not designed as a security measurement,
        // so checksum compute is not done inside ORT.
        last_checksum_validated =
            setting.OptionMatches(
                kNupharCacheModelChecksum,
                std::string(model_checksum, model_checksum_len));

        if (!last_checksum_validated) {
          LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "Cache checksum validation failed, using JIT...";
          disable_caching_due_to_checksum_failure = true;
        }
      } else {
        // do not validate checksum if dll didn't require it (usually during debugging)
        // TODO: force checksum validation in final release
        last_checksum_validated = true;
      }
    }
  }
  return last_checksum_validated;
}

tvm::runtime::PackedFunc LoadTVMPackedFuncFromCache(const std::string& func_name) {
  std::string so_path;
  if (!GetCacheSoFilePath(so_path))
    return nullptr;

  if (!VerifyTVMModuleChecksum(so_path))
    return nullptr;

  tvm::runtime::Module module = tvm::runtime::Module::LoadFromFile(so_path);
  tvm::runtime::PackedFunc func = module.GetFunction(func_name);
  if (func == nullptr) {
    LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "Cannot find " << func_name << " in cache, using JIT...";
  }
  return func;
}

thread_local int saved_tvm_model_cnt = 0;

void SaveTVMModuleToCache(const std::string& filename, tvm::runtime::Module& module) {
  fs::path path;

  if (disable_caching_due_to_checksum_failure)
    return;

  static std::mutex save_cache_mutex;
  static std::unordered_set<std::string> existing_files;
  std::lock_guard<std::mutex> lock(save_cache_mutex);
  if (existing_files.count(filename) == 0 &&
      GetOrCreateTVMModuleCacheDirectory(path, /*create*/ true)) {
    existing_files.insert(filename);
    path.append("cached_" + std::to_string(saved_tvm_model_cnt++) + ".o");
    if (fs::exists(path)) {
      LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "Object file " << path << " already exists, skip saving...";
      return;
    }
    module->SaveToFile(path.string(), "o");
  }
}

std::string GetPackedFuncName(const nuphar::NupharSubgraphUnit& subgraph, const CodeGenTarget& codegen_target) {
  // in C, a function does not allow its name starting with a digit.
  return NormalizeCppName("_" + subgraph.UniqueId() + " " + codegen_target.GetTargetName());
}

}  // namespace nuphar
}  // namespace onnxruntime
