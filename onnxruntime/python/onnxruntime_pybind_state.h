// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_pybind.h"  // must use this for the include of <nanobind/nanobind.h>

namespace onnxruntime {
namespace python {

void addGlobalMethods(py::module& m, Environment& env);
void addObjectMethods(py::module& m, Environment& env);
void addOrtValueMethods(nanobind::module_& m);
void AddLoraMethods(nanobind::module_& m);

}  // namespace python
}  // namespace onnxruntime
