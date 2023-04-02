// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <type_traits>

#include "core/providers/rocm/tunable/gemm_common.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

using BlasOp = onnxruntime::rocm::tunable::blas::BlasOp;

namespace py = pybind11;

namespace onnxruntime {

KE_REGISTER(mod) {
  auto blas_op = mod.def_submodule("blas_op");

  py::enum_<BlasOp>(blas_op, "BlasOp")
      .value("N", BlasOp::N, "Passthrough")
      .value("T", BlasOp::T, "Transpose")
      .export_values();
}

}  // namespace onnxruntime
