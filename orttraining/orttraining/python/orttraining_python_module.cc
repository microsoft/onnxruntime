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

void addGlobalMethods(py::module& m);
void addObjectMethods(py::module& m, ExecutionProviderRegistrationFn ep_registration_fn);
void addObjectMethodsForTraining(py::module& m);
void addObjectMethodsForEager(py::module& m);
#ifdef ENABLE_LAZY_TENSOR
void addObjectMethodsForLazyTensor(py::module& m);
#endif
bool InitArray();

bool GetProviderInstanceHash(const std::string& type,
                             [[maybe_unused]] const ProviderOptionsMap& provider_options_map,
                             size_t& hash) {
  // for built-in execution provider, currently only cpu / cuda support hash.
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
  }

  return false;
}

ORTTrainingPythonEnv::ORTTrainingPythonEnv(OrtEnvPtr ort_env) : ort_env_(std::move(ort_env)) {
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
static OrtThreadingOptions global_tp_options;
static bool use_global_tp = false;

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
  OrtEnvPtr ort_env = OrtEnv::GetOrCreateInstance(lm_info, status, use_global_tp ? &global_tp_options : nullptr);
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

void SetGlobalThreadingOptions(const OrtThreadingOptions&& tp_options) {
  if (ort_training_env != nullptr) {
    OrtPybindThrowIfError(GetEnv().SetGlobalThreadingOptions(tp_options));
  }
  global_tp_options = tp_options;
  use_global_tp = true;
}
bool CheckIfUsingGlobalThreadPool() {
  return use_global_tp;
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
  size_t hash;
  if (GetProviderInstanceHash(provider_type, provider_options_map, hash)) {
    auto cached_provider_instance = training_env.GetExecutionProviderInstance(provider_type, hash);
    if (!cached_provider_instance) {
      auto ep = CreateTrainingEP(session_options, provider_type, provider_options_map);
      if (ep) {
        training_env.AddExecutionProvider(provider_type, hash, std::move(ep));
        cached_provider_instance = training_env.GetExecutionProviderInstance(provider_type, hash);
      }
    }
    return cached_provider_instance;
  } else {
    // the EP doesn't support cache, register the instance to session
    auto ep = CreateTrainingEP(session_options, provider_type, provider_options_map);
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

void CreateQuantPybindModule(py::module& m);

PYBIND11_MODULE(onnxruntime_pybind11_state, m) {
  auto st = CreateTrainingPybindStateModule(m);
  if (!st.IsOK())
    throw pybind11::import_error(st.ErrorMessage());

#ifdef ENABLE_LAZY_TENSOR
  addObjectMethodsForLazyTensor(m);
#endif

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
  CreateQuantPybindModule(m);
}

}  // namespace python
}  // namespace onnxruntime
