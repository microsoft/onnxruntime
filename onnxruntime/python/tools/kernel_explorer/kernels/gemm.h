// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "contrib_ops/rocm/bert/tunable_op.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

namespace py = pybind11;

namespace onnxruntime {

enum class BlasOp {
  N,
  T,
};

inline std::string BlasOpToString(BlasOp op) {
  switch (op) {
    case BlasOp::N:
      return "N";
    case BlasOp::T:
      return "T";
  }
}

// We don't assume the implementation is row-majored or column-majored. But for testing convenience, we assume all
// our wrappers have row-majored convention, since it is the native layout to numpy and pytorch.
template <typename T>
struct GemmParams : contrib::rocm::OpParams {
  std::string Signature() const override {
    return MakeString(BlasOpToString(opa), BlasOpToString(opb), "_", m, "_", n, "_", k);
  }

  rocblas_handle handle;
  BlasOp opa;
  BlasOp opb;
  int64_t m;
  int64_t n;
  int64_t k;
  T alpha;
  T* a;
  int64_t lda;
  T* b;
  int64_t ldb;
  T beta;
  T* c;
  int64_t ldc;
};

void InitGemm(py::module mod);

}  // namespace onnxruntime
