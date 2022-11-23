// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/rocm/gemm_fast_gelu.h"

#include <pybind11/pybind11.h>
#include <type_traits>

#include "core/providers/rocm/tunable/gemm_fast_gelu_common.h"
#include "python/tools/kernel_explorer/kernels/rocm/gemm_fast_gelu_ck.h"
#include "python/tools/kernel_explorer/kernels/rocm/gemm_fast_gelu_unfused.h"
#include "python/tools/kernel_explorer/kernels/rocm/gemm_fast_gelu_tunable.h"

namespace py = pybind11;

namespace onnxruntime {

void InitGemmFastGelu(py::module mod) {
  InitGemmFastGeluUnfused(mod);
  InitComposableKernelGemmFastGelu(mod);
  InitGemmFastGeluTunable(mod);
}

}  // namespace onnxruntime
