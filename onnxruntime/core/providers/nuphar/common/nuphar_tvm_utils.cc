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
#include "core/providers/nuphar/scripts/NUPHAR_CACHE_VERSION"
#include "gsl/gsl"
#include <topi/detail/extern.h>
#include <tvm/ir_pass.h>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING  // required by VS 2019
#include <experimental/filesystem>
#undef _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <atomic>
#include <fstream>
namespace fs = std::experimental::filesystem;

namespace onnxruntime {
namespace nuphar {

static bool GetOrCreateTVMModuleCacheDirectory(fs::path& path, bool create) {
  codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();

  if (!settings.HasOption(kNupharCachePath))
    return false;

  path = settings.GetOptionValue(kNupharCachePath);
  if (!create && !fs::is_directory(path))
    return false;

  if (!fs::is_directory(path))
    if (!fs::create_directory(path)) {
      throw std::runtime_error("Failed to create directory " + path.string());
    }

  path.append(__NUPHAR_CACHE_VERSION__);
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
  ORT_ENFORCE(Env::Default().LoadDynamicLibrary(so_path, false, &so_handle).IsOK());
  void* func = nullptr;
  Status s = Env::Default().GetSymbolFromLibrary(so_handle, func_name, &func);
  if (throw_if_not_found && !s.IsOK())
    ORT_ENFORCE(false, "Cannot find ", func_name, " in ", so_path);
  return func;
}

static void ParseVersion(const char* version, int* major, int* minor, int* patch) {
  std::stringstream ss(version);
  std::string val;

  auto ver_num_fn = [](const std::string& val) {
    ORT_ENFORCE(!val.empty(), "Empty version number");
    if (val.length() > 1 && val[0] == '0') {
      ORT_THROW("Invalid version number: ", val);
    }
    ORT_ENFORCE(std::all_of(val.begin(), val.end(), [](char c) { return isdigit(c); }),
                "Invalid version number: ", val);
    return std::stoi(val);
  };

  std::getline(ss, val, '.');
  ORT_ENFORCE(ss.good(), "Invalid version format: ", version);
  *major = ver_num_fn(val);

  std::getline(ss, val, '.');
  *minor = ver_num_fn(val);

  std::getline(ss, val);
  *patch = ver_num_fn(val);
}

static bool VerifyCacheVersion(const std::string& so_path) {
  // ensure we have _ORTInternal_GetCacheVersion_ function
  void* f = GetFuncFromLibrary(so_path, "_ORTInternal_GetCacheVersion", /*throw_if_not_found*/ true);
  if (f == nullptr)
    return false;

  typedef const char* (*GetVersionFunc)();
  GetVersionFunc func = reinterpret_cast<GetVersionFunc>(f);
  const char* cache_version = func();
  ORT_ENFORCE(cache_version, "Null cache version string!");
  int cur_major, cur_minor, cur_patch;
  ParseVersion(__NUPHAR_CACHE_VERSION__, &cur_major, &cur_minor, &cur_patch);
  int cache_major, cache_minor, cache_patch;
  ParseVersion(cache_version, &cache_major, &cache_minor, &cache_patch);

  // make version check strict until we have thorough design for compatibility
  bool version_match = (cur_major == cache_major) && (cur_minor == cache_minor);
  if (!version_match) {
    LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "Current nuphar runtime version (" << __NUPHAR_CACHE_VERSION__ << ") doesn't match cached dll version (" << cache_version << ")";
  }
  return version_match;
}

static bool VerifyTVMModuleChecksum(const std::string& so_path) {
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
    if (setting.OptionMatches(
            kNupharCacheModelChecksum,
            std::string(model_checksum, model_checksum_len))) {
      return true;
    } else {
      static std::mutex warn_mutex;
      std::lock_guard<std::mutex> warn_lock(warn_mutex);
      static std::string last_warned_so_path;
      if (last_warned_so_path != so_path) {
        // warning only once for each so_path
        LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "Cache checksum validation failed, using JIT...";
      }
      return false;
    }
  } else {
    // do not validate checksum if dll didn't require it (usually during debugging)
    // TODO: force checksum validation in final release
    return true;
  }
}

CacheStatus LoadTVMPackedFuncFromCache(const std::string& func_name, tvm::runtime::PackedFunc& func) {
  std::string so_path;
  if (!GetCacheSoFilePath(so_path)) {
    if (codegen::CodeGenSettings::Instance().HasOption(kNupharCachePath)) {
      return CacheStatus::Missing;
    } else {
      return CacheStatus::NotInUse;
    }
  }

  if (!VerifyCacheVersion(so_path)) {
    return CacheStatus::Mismatch;
  }

  if (!VerifyTVMModuleChecksum(so_path))
    return CacheStatus::Mismatch;

  tvm::runtime::Module module = tvm::runtime::Module::LoadFromFile(so_path);
  func = module.GetFunction(func_name);
  if (func == nullptr) {
    LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "Cannot find " << func_name << " in cache, using JIT...";
    return CacheStatus::Missing;
  }
  return CacheStatus::Found;
}

void SaveTVMModuleToCache(const std::string& filename, tvm::runtime::Module& module) {
  fs::path path;

  static std::mutex save_cache_mutex;
  std::lock_guard<std::mutex> lock(save_cache_mutex);
  if (GetOrCreateTVMModuleCacheDirectory(path, /*create*/ true)) {
    path.append(filename + ".o");
    if (fs::exists(path)) {
      //LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "Object file " << path << " already exists, skip saving...";
      return;
    }
    module->SaveToFile(path.string(), "o");
  }
}

std::string GetPackedFuncName(const nuphar::NupharSubgraphUnit& subgraph, const CodeGenTarget& codegen_target, int64_t parallel_min_workloads) {
  // in C, a function does not allow its name starting with a digit.
  return NormalizeCppName("_" + subgraph.UniqueId() + "_" + codegen_target.GetTargetName() + "_p" + std::to_string(parallel_min_workloads));
}

bool TryCreateConstantScalar(
    tvm::Expr& scalar,
    const Tensor* tensor) {
  if (!tensor)
    return false;

  auto num_elements = tensor->Shape().Size();
  if (num_elements > 1) {
    // for non-scalar, only fold to constant scalar when all values are identical
    const auto& dtype = tensor->DataType();
    auto elem_size = dtype->Size();
    const void* data = tensor->DataRaw();

#define CHECK_ALL_TENSOR_SAME(T)                                                    \
  for (int64_t i = 1; i < num_elements; ++i) {                                      \
    if (reinterpret_cast<const T*>(data)[i] != reinterpret_cast<const T*>(data)[0]) \
      return false;                                                                 \
  }

    switch (elem_size) {
      case 1:
        CHECK_ALL_TENSOR_SAME(int8_t);
        break;
      case 2:
        CHECK_ALL_TENSOR_SAME(int16_t);
        break;
      case 4:
        CHECK_ALL_TENSOR_SAME(int32_t);
        break;
      case 8:
        CHECK_ALL_TENSOR_SAME(int64_t);
        break;
      default:
        return false;
    }

#undef CHECK_ALL_TENSOR_SAME
  }

#define ASSIGN_TVM_SCALAR(tvm_type, tensor_type)                      \
  if (tensor->IsDataType<tensor_type>()) {                            \
    scalar = tvm::make_const(tvm_type, *tensor->Data<tensor_type>()); \
  }

#define ASSIGN_TVM_SCALAR_ELSE(tvm_type, tensor_type) \
  else ASSIGN_TVM_SCALAR(tvm_type, tensor_type)

  ASSIGN_TVM_SCALAR(HalideIR::Float(32), float)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::Float(64), double)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::Int(64), int64_t)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::Int(32), int32_t)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::Int(16), int16_t)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::Int(8), int8_t)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::UInt(64), uint64_t)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::UInt(32), uint32_t)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::UInt(16), uint16_t)
  ASSIGN_TVM_SCALAR_ELSE(HalideIR::UInt(8), uint8_t)
  else {
    return false;
  }

#undef ASSIGN_TVM_SCALAR

  return true;
}

}  // namespace nuphar
}  // namespace onnxruntime
