// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/python/orttraining_pybind_common.h"

#include "core/common/logging/logging.h"
#include "core/common/logging/severity.h"
#include "core/providers/get_execution_providers.h"
#include "core/session/provider_bridge_ort.h"

#ifdef ENABLE_EAGER_MODE
#include "orttraining/python/orttraining_python_module_eager.h"
#endif

namespace onnxruntime {
namespace python {
namespace py = pybind11;

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

void addGlobalMethods(py::module& m, Environment& env);
void addObjectMethods(py::module& m, Environment& env, ExecutionProviderRegistrationFn ep_registration_fn);
void addObjectMethodsForTraining(py::module& m, ExecutionProviderRegistrationFn ep_registration_fn);
void addObjectMethodsForEager(py::module& m);
void InitArray();


bool GetDyanmicExecutionProviderHash(
    const std::string& ep_shared_lib_path,
    const ProviderOptions& provider_options,
    size_t& hash,
    const std::string& entry_symbol_name = "ProviderHashFunc") {
  void* handle;
  auto error = Env::Default().LoadDynamicLibrary(ep_shared_lib_path, false, &handle);
  if (!error.IsOK()) {
    throw std::runtime_error(error.ErrorMessage());
  }

  try{
    size_t (*PGetProviderHash)(const void*) = nullptr;
    OrtPybindThrowIfError(Env::Default().GetSymbolFromLibrary(handle, entry_symbol_name, (void**)&PGetProviderHash));

    if (PGetProviderHash){
      hash = PGetProviderHash(&provider_options);
      return true;
    }
    return false;
  }
  catch(...){
    // there is no ProvideHashFunc provide in the shared lib, which means it doesn't support cache
    return false;
  }
}

bool GetProviderInstanceHash(const std::string& type,
                            const ProviderOptionsMap& provider_options_map,
                            size_t& hash){
  // for built-in execution provider, currently only cpu / cuda / rocm support hash.
  if (type == kCpuExecutionProvider){
    // for CPU, only 1 instance
    hash = 0;
    return true;
  }
  else if (type == kCudaExecutionProvider){
#ifdef USE_CUDA
  if(auto* cuda_provider_info = TryGetProviderInfo_CUDA()){
    const CUDAExecutionProviderInfo info = GetCudaExecutionProviderInfo(cuda_provider_info,
                                                                        provider_options_map);
    hash = static_cast<size_t>(info.device_id) ^
           info. gpu_mem_limit ^
           (static_cast<size_t>(info.arena_extend_strategy) << 16) ^
           (static_cast<size_t>(info.cudnn_conv_algo_search) << 18) ^
           (static_cast<size_t>(info.do_copy_in_default_stream) << 20) ^
           (static_cast<size_t>(info.has_user_compute_stream) << 22);
    return true;
  }
#endif
  }
  else if (type == kRocmExecutionProvider){
#ifdef USE_ROCM
  if(auto* rocm_provider_info = TryGetProviderInfo_ROCM()){
    const ROCMExecutionProviderInfo info = GetRocmExecutionProviderInfo(rocm_provider_info,
                                                                        provider_options_map);
    hash = static_cast<size_t>(info.device_id) ^
           info. gpu_mem_limit ^
           (static_cast<size_t>(info.arena_extend_strategy) << 16) ^
           (static_cast<size_t>(info.miopen_conv_exhaustive_search) << 18) ^
           (static_cast<size_t>(info.do_copy_in_default_stream) << 20) ^
           (static_cast<size_t>(info.has_user_compute_stream) << 22);
    return true;
  }
#endif
  }
  else{
    const auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      auto shared_lib_path_it = it->second.find(kExecutionProviderSharedLibraryPath);
      if (shared_lib_path_it != it->second.end()) {
        // this is an EP with dynamic loading
        // construct the provider option
        ProviderOptions provider_options;
        std::string entry_symbol = kDefaultExecutionProviderEntry;
        for (auto option : it->second) {
          if (option.first == kExecutionProviderSharedLibraryEntry){
            entry_symbol = option.second;
          }
          else if (option.first != kExecutionProviderSharedLibraryPath){
            provider_options.insert(option);
          }
        }
        return GetDyanmicExecutionProviderHash(shared_lib_path_it->second, provider_options, hash);
      }
    }
  }
  return false;
}

ORTTrainingPythonEnv::ORTTrainingPythonEnv(){
  OrtPybindThrowIfError(Environment::Create(std::make_unique<LoggingManager>(
                                                std::make_unique<CLogSink>(),
                                                Severity::kWARNING, false, LoggingManager::InstanceType::Default,
                                                &SessionObjectInitializer::default_logger_id),
                                            ort_env_));
  auto& builtinEPs = GetAvailableExecutionProviderNames();
  available_training_eps_.assign(builtinEPs.begin(), builtinEPs.end());
}

Environment& ORTTrainingPythonEnv::GetORTEnv(){
  return *ort_env_;
}

std::shared_ptr<IExecutionProvider> ORTTrainingPythonEnv::GetExecutionProviderInstance(const std::string& provider_type,
                                                                 size_t hash){
  auto it = execution_provider_instances_map_.find(GetExecutionProviderMapKey(provider_type, hash));
  return it == execution_provider_instances_map_.end() ? nullptr : it->second;
}

void ORTTrainingPythonEnv::AddExecutionProvider(const std::string& provider_type,
                          size_t hash,
                          std::unique_ptr<IExecutionProvider> execution_provider){
  execution_provider_instances_map_.insert({GetExecutionProviderMapKey(provider_type, hash),
                                        std::move(execution_provider)});
}

void ORTTrainingPythonEnv::RegisterExtExecutionProviderInfo(const std::string& provider_type, 
                                      const std::string& provider_lib_path,
                                      const ProviderOptions& default_options){
  ext_execution_provider_info_map_.insert({provider_type, {provider_lib_path, default_options}});
  if (std::find(available_training_eps_.begin(), available_training_eps_.end(), provider_type) == available_training_eps_.end())
    available_training_eps_.push_back(provider_type);
}

const std::vector<std::string>& ORTTrainingPythonEnv::GetAvailableTrainingExecutionProviderTypes(){
  return available_training_eps_;
}

std::string ORTTrainingPythonEnv::GetExecutionProviderMapKey(const std::string& provider_type,
                                       size_t hash){
  std::string key(provider_type);
  key.append(std::to_string(hash));
  return key;
}

void ORTTrainingPythonEnv::ClearExecutionProviderInstances(){
  execution_provider_instances_map_.clear();
}

static std::unique_ptr<ORTTrainingPythonEnv> ort_training_env;

void InitializeTrainingEnv() {
  auto initialize = [&]() {
    static bool initialized = false;
    if (initialized) {
      return;
    }
    // Initialization of the module
    InitArray();
    Env::Default().GetTelemetryProvider().SetLanguageProjection(OrtLanguageProjection::ORT_PROJECTION_PYTHON);
    ort_training_env = std::make_unique<ORTTrainingPythonEnv>();
    initialized = true;
  };
  initialize();
}

ORTTrainingPythonEnv& GetTrainingEnv() {
  if (!ort_training_env) {
    InitializeTrainingEnv();
  }
  return *ort_training_env;
}

Environment& GetTrainingORTEnv() {
  if (!ort_training_env) {
    InitializeTrainingEnv();
  }
  return ort_training_env->GetORTEnv();
}

#ifdef ENABLE_EAGER_MODE
using namespace torch_ort::eager;
static std::unique_ptr<ORTBackendsManager> ort_backends_manager_instance;

void InitializeBackendsManager() {
  auto initialize = [&]() {
    static bool initialized = false;
    if (initialized) {
      return;
    }
    // Initialization of the module
    auto& env = onnxruntime::python::GetTrainingORTEnv();
    ort_backends_manager_instance = std::make_unique<ORTBackendsManager>(env.GetLoggingManager()->DefaultLogger());
    initialized = true;
  };
  initialize();
}

ORTBackendsManager& GetORTBackendsManager() {
  if (!ort_backends_manager_instance) {
    InitializeBackendsManager();
  }
  return *ort_backends_manager_instance;
}
#endif

void ResolveExtraProviderOptions(const std::vector<std::string>& provider_types,
                                 const ProviderOptionsMap& original_provider_options_map,
                                 ProviderOptionsMap& merged_options){
  auto& training_env = GetTrainingEnv();
  for (auto& provider_type : provider_types){
    auto it = training_env.ext_execution_provider_info_map_.find(provider_type);
    if (it == training_env.ext_execution_provider_info_map_.end()){
      //nothing changed.
      if (original_provider_options_map.find(provider_type) != original_provider_options_map.end())
        merged_options.insert({provider_type, original_provider_options_map.at(provider_type)});
    }else{
      ProviderOptions options = it->second.second;
      options.insert({kExecutionProviderSharedLibraryPath, it->second.first});
      auto original_map_it = original_provider_options_map.find(provider_type);
      if (original_map_it != original_provider_options_map.end()){
        for (auto [k, v] : original_map_it->second){
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
  const ProviderOptionsMap& provider_options_map){
  return CreateExecutionProviderInstance(session_options, provider_type, provider_options_map);
}

std::shared_ptr<IExecutionProvider> GetOrCreateExecutionProvider(const std::string& provider_type,
                                                                 const ProviderOptionsMap& provider_options_map,
                                                                 const SessionOptions& session_options){
  ORTTrainingPythonEnv& training_env = GetTrainingEnv();
  // resolve provider options, because the hash key of ep depends on provider options.
  ProviderOptionsMap merged_options;
  ResolveExtraProviderOptions({provider_type}, provider_options_map, merged_options);
  // search in environment
  size_t hash;
  if (GetProviderInstanceHash(provider_type, merged_options, hash)){
    auto cached_provider_instance = training_env.GetExecutionProviderInstance(provider_type, hash);
    if (!cached_provider_instance){
      auto ep = CreateTrainingEP(session_options, provider_type, merged_options);
      if (ep){
        training_env.AddExecutionProvider(provider_type, hash, std::move(ep));
        cached_provider_instance = training_env.GetExecutionProviderInstance(provider_type, hash);
      }
    }
    return cached_provider_instance;
  }
  else{
    // the EP doesn't support cache, register the instance to session
    auto ep = CreateTrainingEP(session_options, provider_type, merged_options);
    return ep;
  }
}

void ORTTrainingRegisterExecutionProviders(InferenceSession* sess, const std::vector<std::string>& provider_types,
                                       const ProviderOptionsMap& provider_options_map) {
  for (auto provider_type : provider_types){
    auto ep = GetOrCreateExecutionProvider(provider_type, provider_options_map, sess->GetSessionOptions());
    if (ep)
      OrtPybindThrowIfError(sess->RegisterExecutionProvider(ep));
  }
}

PYBIND11_MODULE(onnxruntime_pybind11_state, m) {
  m.doc() = "pybind11 stateful interface to ORTTraining";
  RegisterExceptions(m);
  
  Environment& env = GetTrainingORTEnv();
  addGlobalMethods(m, env);
  addObjectMethods(m, env, ORTTrainingRegisterExecutionProviders);
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
  
  addObjectMethodsForTraining(m, ORTTrainingRegisterExecutionProviders);
#ifdef ENABLE_EAGER_MODE
  addObjectMethodsForEager(m);
#endif
  
  m.def("_register_provider_lib", [](const std::string& name, 
                                     const std::string& provider_shared_lib_path,
                                     const ProviderOptions& default_options) {
    GetTrainingEnv().RegisterExtExecutionProviderInfo(name, provider_shared_lib_path, default_options);
  });

  m.def(
      "get_available_providers", []() -> const std::vector<std::string>& { 
        return GetTrainingEnv().GetAvailableTrainingExecutionProviderTypes(); },
      "Return list of available Execution Providers in this installed version of Onnxruntime. "
      "The order of elements represents the default priority order of Execution Providers "
      "from highest to lowest.");
  
  m.def("clear_training_ep_instances", []() -> void {
         ort_training_env->ClearExecutionProviderInstances();
         },
        "Clean the execution provider instances used in ort training module.");

  // clean the ort training environment when python interpreter exit
  // otherwise the global var will be de-constrcut after user main.
  // the order of ort training environment deconstruction and cudart
  // deconstruction is not stable, which will lead to crash.
  auto atexit = py::module_::import("atexit");
  atexit.attr("register")(py::cpp_function([]() {
    ort_training_env = nullptr;
#ifdef ENABLE_EAGER_MODE
    ort_backends_manager_instance = nullptr;
#endif
  }));
}

}
}
