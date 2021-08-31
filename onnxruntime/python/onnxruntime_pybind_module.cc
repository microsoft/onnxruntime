// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>

namespace onnxruntime {
namespace python {
namespace py = pybind11;

void CreateInferencePybindStateModule(py::module& m);

PYBIND11_MODULE(onnxruntime_pybind11_state, m) {
  CreateInferencePybindStateModule(m);
}
}
}