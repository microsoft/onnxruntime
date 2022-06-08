// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "device_array.h"
#include "kernels/vector_add.h"

namespace py = pybind11;

PYBIND11_MODULE(kernel_explorer, m) {
  py::class_<DeviceArray>(m, "DeviceArray")
    .def(py::init<py::array>())
    .def("UpdateHostNumpyArray", &DeviceArray::UpdateHostNumpyArray);
  InitVectorAdd(m);
}
