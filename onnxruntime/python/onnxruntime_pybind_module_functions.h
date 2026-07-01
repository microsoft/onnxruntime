#pragma once
#include "onnxruntime_pybind.h"
#include "core/common/logging/isink.h"
#include "core/framework/provider_options.h"

struct OrtEnv;

namespace onnxruntime {
class InferenceSession;
namespace python {

using ExecutionProviderRegistrationFn = std::function<void(InferenceSession*,
                                                           const std::vector<std::string>&,
                                                           const ProviderOptionsMap&)>;
void addGlobalMethods(pybind11::module& m);
void addObjectMethods(pybind11::module& m, ExecutionProviderRegistrationFn ep_registration_fn);
void addOrtValueMethods(pybind11::module& m);
void addSparseTensorMethods(pybind11::module& m);
void addIoBindingMethods(pybind11::module& m);
void addAdapterFormatMethods(pybind11::module& m);
void addGlobalSchemaFunctions(pybind11::module& m);
void addOpSchemaSubmodule(pybind11::module& m);
void addOpKernelSubmodule(pybind11::module& m);

// Creates the PythonCallbackSink that wraps the platform default sink and can be updated
// later via set_default_logger_callback().  The returned unique_ptr should be passed to
// the LoggingManager; the raw pointer is also stored internally for future updates.
std::unique_ptr<onnxruntime::logging::ISink> CreateAndRegisterPythonCallbackSink(
    std::unique_ptr<onnxruntime::logging::ISink> platform_sink);

// Replaces the Default LoggingManager of ort_env with one backed by a PythonCallbackSink so
// that set_default_logger_callback() can route ORT log messages to a Python callable.  Only
// call this when this Python module created the OrtEnv and before any sessions/threads exist.
void InstallPythonCallbackLoggingSink(OrtEnv& ort_env);

}  // namespace python
}  // namespace onnxruntime
