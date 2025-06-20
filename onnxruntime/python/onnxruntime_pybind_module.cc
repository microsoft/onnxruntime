#include "onnxruntime_pybind_exceptions.h"
#include "onnxruntime_pybind_module_functions.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "core/providers/get_execution_providers.h"
#include "onnxruntime_config.h"
#include "core/common/common.h"
#include "core/session/environment.h"
#include "core/session/ort_env.h"
#include "core/session/inference_session.h"
#include "core/session/provider_bridge_ort.h"
#include "core/framework/provider_options.h"
#include "core/platform/env.h"

// Switch to the 'nb' namespace for nanobind
namespace nb = nanobind;

namespace onnxruntime {
namespace python {

// Forward declarations from other files
std::unique_ptr<IExecutionProvider> CreateExecutionProviderInstance(
    const SessionOptions& session_options,
    const std::string& type,
    const ProviderOptionsMap& provider_options_map);
bool InitArray();
void addGlobalMethods(nanobind::module_& m);
void addObjectMethods(nanobind::module_& m, ExecutionProviderRegistrationFn ep_registration_fn);
void addOrtValueMethods(nanobind::module_& m);
void addSparseTensorMethods(nanobind::module_& m);
void addIoBindingMethods(nanobind::module_& m);
void addAdapterFormatMethods(nanobind::module_& m);
void addGlobalSchemaFunctions(nanobind::module_& m);
void addOpSchemaSubmodule(nanobind::module_& m);
void addOpKernelSubmodule(nanobind::module_& m);
void CreateQuantPybindModule(nanobind::module_& m);  // Updated signature

static OrtEnv* ort_env = nullptr;
static OrtThreadingOptions global_tp_options;
static bool use_global_tp = false;

onnxruntime::Environment& GetEnv() {
  return ort_env->GetEnvironment();
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
    OrtPybindThrowIfError(GetEnv().SetGlobalThreadingOptions(tp_options));
  }
  global_tp_options = tp_options;
  use_global_tp = true;
}

bool CheckIfUsingGlobalThreadPool() {
  return use_global_tp;
}

/**
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

Status CreateInferencePybindStateModule(nanobind::module_& m) {
  // Update the docstring to reflect nanobind
  m.doc() = "nanobind stateful interface to ONNX runtime";
  RegisterExceptions(m);
  if (!InitArray()) {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "import numpy failed");
  }

  ORT_RETURN_IF_ERROR(CreateOrtEnv());

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

// Main module definition
NB_MODULE(onnxruntime_pybind11_state, m) {
  auto st = CreateInferencePybindStateModule(m);
  if (!st.IsOK())
    // Use nanobind's exception type
    throw nanobind::import_error(st.ErrorMessage().c_str());

  m.def("get_available_providers", []() -> const std::vector<std::string>& { return GetAvailableExecutionProviderNames(); },
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