// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernels/vector_add.h"
#include "python/tools/kernel_explorer/kernels/fast_gelu.h"
#include "python/tools/kernel_explorer/kernels/gemm.h"
#include "python/tools/kernel_explorer/kernels/skip_layer_norm.h"

namespace py = pybind11;

namespace onnxruntime {

PYBIND11_MODULE(_kernel_explorer, m) {
  py::class_<DeviceArray>(m, "DeviceArray")
      .def(py::init<py::array>())
      .def("UpdateHostNumpyArray", &DeviceArray::UpdateHostNumpyArray);
  InitVectorAdd(m);
  InitFastGelu(m);
  InitGemm(m);
  InitSkipLayerNorm(m);
}

}  // namespace onnxruntime
