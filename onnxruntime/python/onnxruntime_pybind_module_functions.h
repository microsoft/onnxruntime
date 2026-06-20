#pragma once
#include "onnxruntime_pybind.h"
#include "core/common/logging/isink.h"
#include "core/framework/provider_options.h"

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

}  // namespace python
}  // namespace onnxruntime
