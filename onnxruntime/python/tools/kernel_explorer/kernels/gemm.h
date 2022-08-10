// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/operator.h"

namespace py = pybind11;

namespace onnxruntime {

enum class BlasOp {
  N,
  T,
};

// We don't assume the implementation is row-majored or column-majored. But for testing convenience, we assume all
// our wrappers have row-majored convention, since it is the native layout to numpy and pytorch.
template <typename T>
class GemmBase : public Operator {
 public:
  GemmBase(
      BlasOp opa, BlasOp opb,
      int64_t m, int64_t n, int64_t k,
      double alpha,
      DeviceArray& a, int64_t lda,
      DeviceArray& b, int64_t ldb,
      double beta,
      DeviceArray& c, int64_t ldc)
      : Operator(),
        opa_(opa),
        opb_(opb),
        m_(m),
        n_(n),
        k_(k),
        alpha_(alpha),
        a_(reinterpret_cast<T*>(a.ptr())),
        lda_(lda),
        b_(reinterpret_cast<T*>(b.ptr())),
        ldb_(ldb),
        beta_(beta),
        c_(reinterpret_cast<T*>(c.ptr())),
        ldc_(ldc) {}

  virtual std::vector<std::string> ListImpls() const = 0;
  virtual bool SelectImpl(const std::string& name) = 0;

 protected:
  BlasOp opa_;
  BlasOp opb_;
  int64_t m_;
  int64_t n_;
  int64_t k_;
  T alpha_;
  T* a_;
  int64_t lda_;
  T* b_;
  int64_t ldb_;
  T beta_;
  T* c_;
  int64_t ldc_;
};

void InitGemm(py::module mod);

}  // namespace onnxruntime
