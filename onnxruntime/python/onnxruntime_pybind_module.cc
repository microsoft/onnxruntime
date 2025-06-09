// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_pybind_exceptions.h"
#include "onnxruntime_pybind_module_functions.h"
#include <pybind11/stl.h>
#include "core/providers/get_execution_providers.h"
#include "onnxruntime_config.h"
#include "core/common/common.h"
#include "core/session/ort_env.h"
#include "core/session/inference_session.h"
#include "core/session/provider_bridge_ort.h"
#include "core/framework/provider_options.h"
#include "core/platform/env.h"

namespace onnxruntime {
namespace python {
std::unique_ptr<IExecutionProvider> CreateExecutionProviderInstance(
    const SessionOptions& session_options,
    const std::string& type,
    const ProviderOptionsMap& provider_options_map);
bool InitArray();
static OrtEnv* ort_env = nullptr;
static OrtThreadingOptions global_tp_options;
static bool use_global_tp = false;
onnxruntime::Environment& GetEnv() {
  return ort_env->GetEnvironment();
}
bool IsOrtEnvInitialized() {
  return ort_env != nullptr;
}
OrtEnv* GetOrtEnv() {
  return ort_env;
}
static Status CreateOrtEnv() {
  Env::Default().GetTelemetryProvider().SetLanguageProjection(OrtLanguageProjection::ORT_PROJECTION_PYTHON);
  OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr, ORT_LOGGING_LEVEL_WARNING, "Default"};
  Status status;
  ort_env = OrtEnv::GetInstance(lm_info, status, use_global_tp ? &global_tp_options : nullptr);
  if (!status.IsOK()) return status;
  // Keep the ort_env alive, don't free it. It's ok to leak the memory.
#if !defined(__APPLE__) && !defined(ORT_MINIMAL_BUILD)
  if (!InitProvidersSharedLibrary()) {
    const logging::Logger& default_logger = ort_env->GetLoggingManager()->DefaultLogger();
    LOGS(default_logger, WARNING) << "Init provider bridge failed.";
  }
#endif
  return Status::OK();
}

void SetGlobalThreadingOptions(const OrtThreadingOptions&& tp_options) {
  if (ort_env != nullptr) {
    ORT_THROW("Global threading options can only be set before the environment is initialized.");
  }
  global_tp_options = tp_options;
  use_global_tp = true;
}
bool CheckIfUsingGlobalThreadPool() {
  return use_global_tp;
}

namespace py = pybind11;

/*
 * Register execution provider with options.
 */
static void RegisterExecutionProviders(InferenceSession* sess, const std::vector<std::string>& provider_types,
                                       const ProviderOptionsMap& provider_options_map) {
  for (const std::string& type : provider_types) {
    auto ep = CreateExecutionProviderInstance(sess->GetSessionOptions(), type, provider_options_map);
    if (ep) {
      OrtPybindThrowIfError(sess->RegisterExecutionProvider(std::move(ep)));
    }
  }
}

Status CreateInferencePybindStateModule(py::module& m) {
  m.doc() = "pybind11 stateful interface to ONNX runtime";
  RegisterExceptions(m);
  if (!InitArray()) {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "import numpy failed");
  }

  addGlobalMethods(m);
  addObjectMethods(m, RegisterExecutionProviders);
  addOrtValueMethods(m);
  addSparseTensorMethods(m);
  addIoBindingMethods(m);
  addAdapterFormatMethods(m);
  addGlobalSchemaFunctions(m);
  addOpSchemaSubmodule(m);
  addOpKernelSubmodule(m);
  return Status::OK();
}

#if defined(USE_MPI) && defined(ORT_USE_NCCL)
static constexpr bool HAS_COLLECTIVE_OPS = true;
#else
static constexpr bool HAS_COLLECTIVE_OPS = false;
#endif

void CreateQuantPybindModule(py::module& m);

PYBIND11_MODULE(onnxruntime_pybind11_state, m) {
  auto st = CreateInferencePybindStateModule(m);
  if (!st.IsOK())
    throw pybind11::import_error(st.ErrorMessage());
  // move it out of shared method since training build has a little different behavior.
  m.def(
      "get_available_providers", []() -> const std::vector<std::string>& { return GetAvailableExecutionProviderNames(); },
      "Return list of available Execution Providers in this installed version of Onnxruntime. "
      "The order of elements represents the default priority order of Execution Providers "
      "from highest to lowest.");

  m.def("get_version_string", []() -> std::string { return ORT_VERSION; });
  m.def("get_build_info", []() -> std::string { return ORT_BUILD_INFO; });
  m.def("has_collective_ops", []() -> bool { return HAS_COLLECTIVE_OPS; });
  CreateQuantPybindModule(m);
}
}  // namespace python
}  // namespace onnxruntime
