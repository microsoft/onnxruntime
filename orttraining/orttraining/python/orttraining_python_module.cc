// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/python/orttraining_pybind_common.h"
#include "python/onnxruntime_pybind_mlvalue.h"

#include "core/common/logging/logging.h"
#include "core/common/logging/severity.h"
#include "core/common/path_string.h"
#include "core/providers/get_execution_providers.h"
#include "core/session/provider_bridge_ort.h"
#include "onnxruntime_config.h"

namespace onnxruntime {
namespace python {
namespace py = pybind11;

#if defined(USE_MPI) && defined(ORT_USE_NCCL)
static constexpr bool HAS_COLLECTIVE_OPS = true;
#else
static constexpr bool HAS_COLLECTIVE_OPS = false;
#endif

using namespace onnxruntime::logging;

std::unique_ptr<IExecutionProvider> CreateExecutionProviderInstance(
    const SessionOptions& session_options,
    const std::string& type,
    const ProviderOptionsMap& provider_options_map);

#ifdef USE_CUDA
const CUDAExecutionProviderInfo GetCudaExecutionProviderInfo(ProviderInfo_CUDA* cuda_provider_info,
                                                             const ProviderOptionsMap& provider_options_map);
#endif

#ifdef USE_ROCM
const ROCMExecutionProviderInfo GetRocmExecutionProviderInfo(ProviderInfo_ROCM* rocm_provider_info,
                                                             const ProviderOptionsMap& provider_options_map);
#endif

void addGlobalMethods(py::module& m);
void addObjectMethods(py::module& m, ExecutionProviderRegistrationFn ep_registration_fn);
void addObjectMethodsForTraining(py::module& m);
void addObjectMethodsForEager(py::module& m);
#ifdef ENABLE_LAZY_TENSOR
void addObjectMethodsForLazyTensor(py::module& m);
#endif
bool InitArray();

bool GetDynamicExecutionProviderHash(
    const std::string& ep_shared_lib_path,
    const ProviderOptions& provider_options,
    size_t& hash,
    const std::string& entry_symbol_name = "ProviderHashFunc") {
  void* handle;
  const auto path_str = ToPathString(ep_shared_lib_path);
  auto error = Env::Default().LoadDynamicLibrary(path_str, false, &handle);
  if (!error.IsOK()) {
    throw std::runtime_error(error.ErrorMessage());
  }

  try {
    size_t (*PGetProviderHash)(const void*) = nullptr;
    OrtPybindThrowIfError(Env::Default().GetSymbolFromLibrary(handle, entry_symbol_name, (void**)&PGetProviderHash));

    if (PGetProviderHash) {
      hash = PGetProviderHash(&provider_options);
      return true;
    }
    return false;
  } catch (...) {
    // there is no ProvideHashFunc provide in the shared lib, which means it doesn't support cache
    return false;
  }
}

bool GetProviderInstanceHash(const std::string& type,
                             const ProviderOptionsMap& provider_options_map,
                             size_t& hash) {
  // for built-in execution provider, currently only cpu / cuda / rocm support hash.
  if (type == kCpuExecutionProvider) {
    // for CPU, only 1 instance
    hash = 0;
    return true;
  } else if (type == kCudaExecutionProvider) {
#ifdef USE_CUDA
    if (auto* cuda_provider_info = TryGetProviderInfo_CUDA()) {
      const CUDAExecutionProviderInfo info = GetCudaExecutionProviderInfo(cuda_provider_info,
                                                                          provider_options_map);
      hash = std::hash<CUDAExecutionProviderInfo>{}(info);
      return true;
    }
#endif
  } else if (type == kRocmExecutionProvider) {
#ifdef USE_ROCM
    if (auto* rocm_provider_info = TryGetProviderInfo_ROCM()) {
      const ROCMExecutionProviderInfo info = GetRocmExecutionProviderInfo(rocm_provider_info,
                                                                          provider_options_map);
      hash = std::hash<ROCMExecutionProviderInfo>{}(info);
      return true;
    }
#endif
  } else {
    const auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      auto shared_lib_path_it = it->second.find(kExecutionProviderSharedLibraryPath);
      if (shared_lib_path_it != it->second.end()) {
        // this is an EP with dynamic loading
        // construct the provider option
        ProviderOptions provider_options;
        std::string entry_symbol = kDefaultExecutionProviderEntry;
        for (auto option : it->second) {
          if (option.first == kExecutionProviderSharedLibraryEntry) {
            entry_symbol = option.second;
          } else if (option.first != kExecutionProviderSharedLibraryPath) {
            provider_options.insert(option);
          }
        }
        return GetDynamicExecutionProviderHash(shared_lib_path_it->second, provider_options, hash);
      }
    }
  }
  return false;
}

ORTTrainingPythonEnv::ORTTrainingPythonEnv(std::unique_ptr<OrtEnv> ort_env) : ort_env_(std::move(ort_env)) {
  const auto& builtinEPs = GetAvailableExecutionProviderNames();
  available_training_eps_.assign(builtinEPs.begin(), builtinEPs.end());
}

const OrtEnv& ORTTrainingPythonEnv::GetORTEnv() const {
  return *ort_env_;
}

OrtEnv& ORTTrainingPythonEnv::GetORTEnv() {
  return *ort_env_;
}

std::shared_ptr<IExecutionProvider> ORTTrainingPythonEnv::GetExecutionProviderInstance(const std::string& provider_type,
                                                                                       size_t hash) {
  auto it = execution_provider_instances_map_.find(GetExecutionProviderMapKey(provider_type, hash));
  return it == execution_provider_instances_map_.end() ? nullptr : it->second;
}

void ORTTrainingPythonEnv::AddExecutionProvider(const std::string& provider_type,
                                                size_t hash,
                                                std::unique_ptr<IExecutionProvider> execution_provider) {
  execution_provider_instances_map_.insert({GetExecutionProviderMapKey(provider_type, hash),
                                            std::move(execution_provider)});
}

void ORTTrainingPythonEnv::RegisterExtExecutionProviderInfo(const std::string& provider_type,
                                                            const std::string& provider_lib_path,
                                                            const ProviderOptions& default_options) {
  ext_execution_provider_info_map_.insert({provider_type, {provider_lib_path, default_options}});
  if (std::find(available_training_eps_.begin(), available_training_eps_.end(), provider_type) == available_training_eps_.end())
    available_training_eps_.push_back(provider_type);
}

const std::vector<std::string>& ORTTrainingPythonEnv::GetAvailableTrainingExecutionProviderTypes() {
  return available_training_eps_;
}

std::string ORTTrainingPythonEnv::GetExecutionProviderMapKey(const std::string& provider_type,
                                                             size_t hash) {
  std::string key(provider_type);
  key.append(std::to_string(hash));
  return key;
}

void ORTTrainingPythonEnv::ClearExecutionProviderInstances() {
  execution_provider_instances_map_.clear();
}

static ORTTrainingPythonEnv* ort_training_env = nullptr;

OrtEnv* GetOrtEnv() {
  return &ort_training_env->GetORTEnv();
}
onnxruntime::Environment& GetEnv() {
  return ort_training_env->GetORTEnv().GetEnvironment();
}

static Status CreateOrtEnv() {
  Env::Default().GetTelemetryProvider().SetLanguageProjection(OrtLanguageProjection::ORT_PROJECTION_PYTHON);
  OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr, ORT_LOGGING_LEVEL_WARNING, "Default"};
  Status status;
  std::unique_ptr<OrtEnv> ort_env(OrtEnv::GetInstance(lm_info, status));
  if (!status.IsOK()) return status;
#if !defined(__APPLE__) && !defined(ORT_MINIMAL_BUILD)
  if (!InitProvidersSharedLibrary()) {
    const logging::Logger& default_logger = ort_env->GetLoggingManager()->DefaultLogger();
    LOGS(default_logger, WARNING) << "Init provider bridge failed.";
  }
#endif
  ort_training_env = new ORTTrainingPythonEnv(std::move(ort_env));
  return Status::OK();
}

ORTTrainingPythonEnv& GetTrainingEnv() {
  return *ort_training_env;
}

void ResolveExtraProviderOptions(const std::vector<std::string>& provider_types,
                                 const ProviderOptionsMap& original_provider_options_map,
                                 ProviderOptionsMap& merged_options) {
  auto& training_env = GetTrainingEnv();
  for (auto& provider_type : provider_types) {
    auto it = training_env.ext_execution_provider_info_map_.find(provider_type);
    if (it == training_env.ext_execution_provider_info_map_.end()) {
      // nothing changed.
      if (original_provider_options_map.find(provider_type) != original_provider_options_map.end())
        merged_options.insert({provider_type, original_provider_options_map.at(provider_type)});
    } else {
      ProviderOptions options = it->second.second;
      options.insert({kExecutionProviderSharedLibraryPath, it->second.first});
      auto original_map_it = original_provider_options_map.find(provider_type);
      if (original_map_it != original_provider_options_map.end()) {
        for (auto [k, v] : original_map_it->second) {
          options.insert({k, v});
        }
      }
      merged_options[provider_type] = options;
    }
  }
}

std::unique_ptr<IExecutionProvider> CreateTrainingEP(
    const SessionOptions& session_options,
    const std::string& provider_type,
    const ProviderOptionsMap& provider_options_map) {
  // TODO(leca): REVIEW: No allocators are initialized
  return CreateExecutionProviderInstance(session_options, provider_type, provider_options_map);
}

std::shared_ptr<IExecutionProvider> GetOrCreateExecutionProvider(const std::string& provider_type,
                                                                 const ProviderOptionsMap& provider_options_map,
                                                                 const SessionOptions& session_options) {
  ORTTrainingPythonEnv& training_env = GetTrainingEnv();
  // resolve provider options, because the hash key of ep depends on provider options.
  ProviderOptionsMap merged_options;
  ResolveExtraProviderOptions({provider_type}, provider_options_map, merged_options);
  // search in environment
  size_t hash;
  if (GetProviderInstanceHash(provider_type, merged_options, hash)) {
    auto cached_provider_instance = training_env.GetExecutionProviderInstance(provider_type, hash);
    if (!cached_provider_instance) {
      auto ep = CreateTrainingEP(session_options, provider_type, merged_options);
      if (ep) {
        training_env.AddExecutionProvider(provider_type, hash, std::move(ep));
        cached_provider_instance = training_env.GetExecutionProviderInstance(provider_type, hash);
      }
    }
    return cached_provider_instance;
  } else {
    // the EP doesn't support cache, register the instance to session
    auto ep = CreateTrainingEP(session_options, provider_type, merged_options);
    return ep;
  }
}

void ORTTrainingRegisterExecutionProviders(InferenceSession* sess, const std::vector<std::string>& provider_types,
                                           const ProviderOptionsMap& provider_options_map) {
  for (auto provider_type : provider_types) {
    auto ep = GetOrCreateExecutionProvider(provider_type, provider_options_map, sess->GetSessionOptions());
    if (ep)
      OrtPybindThrowIfError(sess->RegisterExecutionProvider(ep));
  }
}

Status CreateTrainingPybindStateModule(py::module& m) {
  m.doc() = "pybind11 stateful interface to ORTTraining";
  RegisterExceptions(m);
  if (!InitArray()) {
    return Status(::onnxruntime::common::ONNXRUNTIME, ::onnxruntime::common::FAIL, "import numpy failed");
  }
  ORT_RETURN_IF_ERROR(CreateOrtEnv());

  addGlobalMethods(m);
  addObjectMethods(m, ORTTrainingRegisterExecutionProviders);
  addOrtValueMethods(m);
  addSparseTensorMethods(m);
  addIoBindingMethods(m);
  addAdapterFormatMethods(m);
  addObjectMethodsForTraining(m);

  return Status::OK();
}

PYBIND11_MODULE(onnxruntime_pybind11_state, m) {
  auto st = CreateTrainingPybindStateModule(m);
  if (!st.IsOK())
    throw pybind11::import_error(st.ErrorMessage());

#ifdef ENABLE_LAZY_TENSOR
  addObjectMethodsForLazyTensor(m);
#endif

  m.def("_register_provider_lib", [](const std::string& name,
                                     const std::string& provider_shared_lib_path,
                                     const ProviderOptions& default_options) {
    GetTrainingEnv().RegisterExtExecutionProviderInfo(name, provider_shared_lib_path, default_options);
  });

  m.def(
      "get_available_providers", []() -> const std::vector<std::string>& { return GetTrainingEnv().GetAvailableTrainingExecutionProviderTypes(); },
      "Return list of available Execution Providers in this installed version of Onnxruntime. "
      "The order of elements represents the default priority order of Execution Providers "
      "from highest to lowest.");

  m.def("get_version_string", []() -> std::string { return ORT_VERSION; });

  m.def("get_build_info", []() -> std::string { return ORT_BUILD_INFO; });

  m.def(
      "clear_training_ep_instances", []() -> void {
        GetTrainingEnv().ClearExecutionProviderInstances();
      },
      "Clean the execution provider instances used in ort training module.");

  m.def("has_collective_ops", []() -> bool { return HAS_COLLECTIVE_OPS; });
}

}  // namespace python
}  // namespace onnxruntime
