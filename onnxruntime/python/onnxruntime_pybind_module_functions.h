#pragma once
#include "onnxruntime_pybind.h"
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
}  // namespace python
}  // namespace onnxruntime