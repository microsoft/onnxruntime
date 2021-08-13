// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_state_common.h"

#include "core/common/logging/logging.h"
#include "core/common/logging/severity.h"

#include "core/platform/env.h"
#include "core/session/provider_bridge_ort.h"

#include <unordered_map>

namespace onnxruntime {
namespace python {
namespace py = pybind11;

using namespace onnxruntime::logging;

std::unique_ptr<IExecutionProvider> CreateExecutionProviderInstance(
  InferenceSession* sess,
  const std::string& type,
  const ProviderOptionsMap& provider_options_map);

void addGlobalMethods(py::module& m, Environment& env);
void addObjectMethods(py::module& m, Environment& env, ExecutionProviderRegistrationFn ep_registration_fn);
void addObjectMethodsForTraining(py::module& m, ExecutionProviderRegistrationFn ep_registration_fn);
void addObjectMethodsForEager(py::module& m);
void InitArray();

using ExecutionProviderMap = std::unordered_map<std::string, std::shared_ptr<IExecutionProvider> >;

class ORTTrainingPythonEnv{
public:
  ORTTrainingPythonEnv(){
    OrtPybindThrowIfError(Environment::Create(std::make_unique<LoggingManager>(
                                                  std::unique_ptr<ISink>{new CLogSink{}},
                                                  Severity::kWARNING, false, LoggingManager::InstanceType::Default,
                                                  &SessionObjectInitializer::default_logger_id),
                                              ort_env_));
  }

  Environment& GetORTEnv(){
    return *ort_env_;
  }

  std::shared_ptr<IExecutionProvider> GetExecutionProviderInstance(const std::string& provider_type,
                                                                   const ProviderOptions& provider_options){
    auto it = execution_provider_instances_.find(GetExecutionProviderMapKey(provider_type, provider_options));
    return it == execution_provider_instances_.end() ? nullptr : it->second;
  }

  void AddExecutionProvider(const std::string& provider_type,
                            const ProviderOptions& provider_options,
                            std::unique_ptr<IExecutionProvider> execution_provider){
    execution_provider_instances_.insert({GetExecutionProviderMapKey(provider_type, provider_options),
                                          std::move(execution_provider)});
  }

private:
  std::string GetExecutionProviderMapKey(const std::string& provider_type,
                                         const ProviderOptions& provider_options){
    std::string key(provider_type);
    //TODO: there is some EP factory method doesn't take a look at provider_options at all
    //For that case, same provider type with different provider options should point to the same instance
    for (auto option : provider_options){
      key.append(1, ' ').append(option.first).append(1, ':').append(option.second);
    }
    return key;
  }

  std::unique_ptr<Environment> ort_env_;
  ExecutionProviderMap execution_provider_instances_;
};

static std::unique_ptr<ORTTrainingPythonEnv> ort_training_env;

void InitializeTrainingEnv() {
  auto initialize = [&]() {
    // Initialization of the module
    InitArray();
    Env::Default().GetTelemetryProvider().SetLanguageProjection(OrtLanguageProjection::ORT_PROJECTION_PYTHON);
    ort_training_env = std::make_unique<ORTTrainingPythonEnv>();
    static bool initialized = false;
    if (initialized) {
      return;
    }
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

void ORTTrainingRegisterExecutionProviders(InferenceSession* sess, const std::vector<std::string>& provider_types,
                                       const ProviderOptionsMap& provider_options_map) {
  // search in environment
  ORTTrainingPythonEnv& training_env = GetTrainingEnv();
  for (auto provider_type : provider_types){
    auto it = provider_options_map.find(provider_type);
    auto cached_provider_instance = training_env.GetExecutionProviderInstance(provider_type,
                                      it == provider_options_map.end() ? ProviderOptions{} : it->second);
    if (!cached_provider_instance){
      auto ep = CreateExecutionProviderInstance(sess, provider_type, provider_options_map);
      if (ep){
        training_env.AddExecutionProvider(provider_type,
                                          it == provider_options_map.end() ? ProviderOptions{} : it->second,
                                          std::move(ep));
        cached_provider_instance = training_env.GetExecutionProviderInstance(provider_type,
                                    it == provider_options_map.end() ? ProviderOptions{} : it->second);
      }
    }
    if (cached_provider_instance)
      OrtPybindThrowIfError(sess->RegisterExecutionProvider(cached_provider_instance));
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
}

}
}