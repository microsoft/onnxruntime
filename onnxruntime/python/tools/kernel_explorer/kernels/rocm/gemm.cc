// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/rocm/gemm.h"

#include <pybind11/pybind11.h>
#include <type_traits>

#include "core/providers/rocm/tunable/gemm_common.h"
#include "python/tools/kernel_explorer/kernels/rocm/gemm_ck.h"
#include "python/tools/kernel_explorer/kernels/rocm/gemm_rocblas.h"
#include "python/tools/kernel_explorer/kernels/rocm/gemm_tunable.h"

using BlasOp = onnxruntime::rocm::tunable::blas::BlasOp;

namespace py = pybind11;

namespace onnxruntime {

void InitGemm(py::module mod) {
  auto blas_op = mod.def_submodule("blas_op");

  py::enum_<BlasOp>(blas_op, "BlasOp")
      .value("N", BlasOp::N, "Passthrough")
      .value("T", BlasOp::T, "Transpose")
      .export_values();

  InitRocBlasGemm(mod);
  InitComposableKernelGemm(mod);
  InitTunableGemm(mod);
}

}  // namespace onnxruntime
