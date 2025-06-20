#pragma once
#include "onnxruntime_pybind.h"
#include <functional>
#include "core/framework/provider_options.h"

namespace onnxruntime {
class InferenceSession;
namespace python {

using ExecutionProviderRegistrationFn = std::function<void(InferenceSession*,
                                                           const std::vector<std::string>&,
                                                           const ProviderOptionsMap&)>;
void addGlobalMethods(nanobind::module_& m);
void addObjectMethods(nanobind::module_& m, ExecutionProviderRegistrationFn ep_registration_fn);
void addOrtValueMethods(nanobind::module_& m);
void addSparseTensorMethods(nanobind::module_& m);
void addIoBindingMethods(nanobind::module_& m);
void addAdapterFormatMethods(nanobind::module_& m);
void addGlobalSchemaFunctions(nanobind::module_& m);
void addOpSchemaSubmodule(nanobind::module_& m);
void addOpKernelSubmodule(nanobind::module_& m);
}  // namespace python
}  // namespace onnxruntime