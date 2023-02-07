// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernels/vector_add.h"
#include "python/tools/kernel_explorer/kernels/rocm/fast_gelu.h"
#include "python/tools/kernel_explorer/kernels/rocm/gemm.h"
#include "python/tools/kernel_explorer/kernels/rocm/skip_layer_norm.h"
#include "python/tools/kernel_explorer/kernels/rocm/gemm_fast_gelu.h"

namespace py = pybind11;

namespace onnxruntime {

PYBIND11_MODULE(_kernel_explorer, m) {
  py::class_<DeviceArray>(m, "DeviceArray")
      .def(py::init<py::array>())
      .def("UpdateHostNumpyArray", &DeviceArray::UpdateHostNumpyArray);
  InitVectorAdd(m);
#if USE_ROCM
  InitFastGelu(m);
  InitGemm(m);
  InitSkipLayerNorm(m);
  InitGemmFastGelu(m);
#endif

  m.def("is_composable_kernel_available", []() {
#ifdef USE_COMPOSABLE_KERNEL
    return true;
#else
    return false;
#endif
  });
}

}  // namespace onnxruntime
