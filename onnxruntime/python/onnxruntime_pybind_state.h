// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_pybind.h"  // must use this for the include of <pybind11/pybind11.h>

namespace onnxruntime {
namespace python {

void addGlobalMethods(py::module& m, Environment& env);
void addObjectMethods(py::module& m, Environment& env);
void addOrtValueMethods(pybind11::module& m);

}  // namespace python
}  // namespace onnxruntime
