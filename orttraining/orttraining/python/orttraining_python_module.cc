// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include <numpy/arrayobject.h>

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

bool GetDyanmicExecutionProviderHash(
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
      hash = static_cast<size_t>(info.device_id) ^
             info.gpu_mem_limit ^
             (static_cast<size_t>(info.arena_extend_strategy) << 16) ^
             (static_cast<size_t>(info.cudnn_conv_algo_search) << 18) ^
             (static_cast<size_t>(info.do_copy_in_default_stream) << 20) ^
             (static_cast<size_t>(info.has_user_compute_stream) << 22) ^
             std::hash<cuda::TunableOpInfo>{}(info.tunable_op);
      return true;
    }
#endif
  } else if (type == kRocmExecutionProvider) {
#ifdef USE_ROCM
    if (auto* rocm_provider_info = TryGetProviderInfo_ROCM()) {
      const ROCMExecutionProviderInfo info = GetRocmExecutionProviderInfo(rocm_provider_info,
                                                                          provider_options_map);
      hash = static_cast<size_t>(info.device_id) ^
             info.gpu_mem_limit ^
             (static_cast<size_t>(info.arena_extend_strategy) << 16) ^
             (static_cast<size_t>(info.miopen_conv_exhaustive_search) << 18) ^
             (static_cast<size_t>(info.do_copy_in_default_stream) << 20) ^
             (static_cast<size_t>(info.has_user_compute_stream) << 22) ^
             std::hash<rocm::TunableOpInfo>{}(info.tunable_op);
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
        return GetDyanmicExecutionProviderHash(shared_lib_path_it->second, provider_options, hash);
      }
    }
  }
  return false;
}

ORTTrainingPythonEnv::ORTTrainingPythonEnv() : ort_env_(GetEnv()) {
  const auto& builtinEPs = GetAvailableExecutionProviderNames();
  available_training_eps_.assign(builtinEPs.begin(), builtinEPs.end());
}

std::shared_ptr<Environment> ORTTrainingPythonEnv::GetORTEnv() const {
  return ort_env_;
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

namespace {

// This class provides a static shell for on-demand and thread-safe construction
// of ORTTrainingPythonEnv object for both Inference and Training python layers.
// ORTTrainingPythonEnv class contains instances of execution providers that have been
// instantiated for training purposes. It depends on the Environment singleton to which it
// holds a shared_ptr instance.
//
// 1) we make this class a singleton that is a function local static. The function local statics
//    are constructed when the function is called the very first time. This fact has several important
//    properties.
//    - First, it is constructed before it is first needed possibly by another static object
//      and destroyed after that object is destroyed.
//    - Second, it is constructed in a thread safe manner.
//    - Last, this order of construction/destruction is enforced across the compilation units, as opposed
//      to the static objects that are simply declared in order in a single unit, but their lifespan is
//      unconnected to that of in other compilation units. This is achieved automatically by run-time
//      by execution atexit() to build a chain.
// 2) This ORTTrainingPythonEnv is currently owned by a unique_ptr unlike the Environment singleton. This is
//    because we currently do not see a need to refer to it by any of the Python objects or by other singletons.
//    With this change this singleton is properly destroyed after python module is unloaded, but before the Environment.
//    HOWEVER, because it holds instances of execution providers, we want to make sure that those instances are destroyed
//    before those depended EP DLLs are unloaded so EP destructor can run.
//    This static is destroyed when this compilation unit is unloaded and it generally happens
//    AFTER EP dlls are unloaded. To mitigate that, we clear EP instances using python `atexit` (different from C atexit())
//    mechanism which takes place after all python objects are GCed but before any DLLs are unloaded or
//    runtime starts destroying globals.
// 3) We guard against singleton resurrection attempts to detect code that runs when it should not
//    and make necessary adjustments.
//    For all the related details and why it is needed see "Modern C++ design" by A. Alexandrescu Chapter 6.
class TrainingEnvInitialzer {
 public:
  static ORTTrainingPythonEnv& Instance() {
    // Guard against attempts to resurrect the singleton
    if (TrainingEnvInitialzer::destroyed) {
      ORT_THROW("Detected an attempt to resurrect destroyed Training Environment");
    }

    static TrainingEnvInitialzer training_env_holder;

    return training_env_holder.Get();
  }

 private:
  TrainingEnvInitialzer() {
    Env::Default().GetTelemetryProvider().SetLanguageProjection(OrtLanguageProjection::ORT_PROJECTION_PYTHON);
    ort_training_env_ = std::make_unique<ORTTrainingPythonEnv>();
  }

  ~TrainingEnvInitialzer() {
    destroyed = true;
  }

  ORTTrainingPythonEnv& Get() noexcept {
    return *ort_training_env_;
  }

  std::unique_ptr<ORTTrainingPythonEnv> ort_training_env_;

  static bool destroyed;
};

bool TrainingEnvInitialzer::destroyed = false;

}  // namespace

ORTTrainingPythonEnv& GetTrainingEnv() {
  return TrainingEnvInitialzer::Instance();
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

static bool CreateTrainingPybindStateModule(py::module& m) {
  import_array1(false);
  m.doc() = "pybind11 stateful interface to ORTTraining";
  RegisterExceptions(m);

  // Instantiate singletons
  GetTrainingEnv();
  addGlobalMethods(m);
  addObjectMethods(m, ORTTrainingRegisterExecutionProviders);
  addOrtValueMethods(m);
  addSparseTensorMethods(m);
  addIoBindingMethods(m);

#if !defined(__APPLE__) && \
    (!defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS))
  Ort::SessionOptions tmp_options;
  if (!InitProvidersSharedLibrary()) {
    const logging::Logger& default_logger = logging::LoggingManager::DefaultLogger();
    LOGS(default_logger, WARNING) << "Init provider bridge failed.";
  }
#endif

  addObjectMethodsForTraining(m);

#ifdef ENABLE_LAZY_TENSOR
  addObjectMethodsForLazyTensor(m);
#endif
}
PYBIND11_MODULE(onnxruntime_pybind11_state, m) {
  if (!CreateTrainingPybindStateModule(m)) {
    throw pybind11::import_error();
  }

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

  // See documentation for class TrainingEnvInitialzer earlier in this module
  // for an explanation as to why this is needed.
  auto atexit = py::module_::import("atexit");
  atexit.attr("register")(py::cpp_function([]() {
    GetTrainingEnv().ClearExecutionProviderInstances();
  }));
}

}  // namespace python
}  // namespace onnxruntime
