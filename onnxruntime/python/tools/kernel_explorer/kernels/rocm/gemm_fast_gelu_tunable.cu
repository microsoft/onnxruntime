// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/rocm/gemm_fast_gelu_tunable.h"

#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>

#include "core/providers/rocm/tunable/gemm_fast_gelu_common.h"
#include "core/providers/rocm/tunable/gemm_fast_gelu_tunable.cuh"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

using namespace onnxruntime::rocm::tunable::blas;
using namespace onnxruntime::rocm::tunable::blas::internal;

namespace py = pybind11;

namespace onnxruntime {

template <typename T>
class GemmFastGeluUnfused : public IKernelExplorer {
 public:
  GemmFastGeluUnfused(BlasOp opa, BlasOp opb,
                      int64_t m, int64_t n, int64_t k,
                      double alpha,
                      DeviceArray& a, int64_t lda,
                      DeviceArray& b, int64_t ldb,
                      DeviceArray& bias,
                      double beta,
                      DeviceArray& c, int64_t ldc) : params_{} {
    ROCBLAS_CALL_THROW(rocblas_create_handle(&rocblas_handle_));
    params_.tuning = true;
    params_.stream = Stream();
    params_.handle = rocblas_handle_;
    params_.opa = opa;
    params_.opb = opb;
    params_.m = m;
    params_.n = n;
    params_.k = k;
    params_.alpha = alpha;
    params_.a = static_cast<T*>(a.ptr());
    params_.lda = lda;
    params_.b = static_cast<T*>(b.ptr());
    params_.ldb = ldb;
    params_.bias = static_cast<T*>(bias.ptr());
    params_.beta = beta;
    params_.c = static_cast<T*>(c.ptr());
    params_.ldc = ldc;
  }

  ~GemmFastGeluUnfused() {
    ROCBLAS_CALL_THROW(rocblas_destroy_handle(rocblas_handle_));
    rocblas_handle_ = nullptr;
  }

  void Run() override {
    ORT_THROW_IF_ERROR(rocm::tunable::blas::internal::GemmFastGeluUnfused<T>(&params_));
  }

  bool IsSupported() {
    Status status = rocm::tunable::blas::internal::GemmFastGeluUnfused<T>(&params_);
    return status.IsOK();
  }

 private:
  using ParamsT = GemmFastGeluParams<T>;
  ParamsT params_{};
  rocblas_handle rocblas_handle_;
};

template <typename T, typename ALayout, typename BLayout>
class GemmFastGeluTunable : public IKernelExplorer {
 public:
  GemmFastGeluTunable(BlasOp opa, BlasOp opb,
                      int64_t m, int64_t n, int64_t k,
                      double alpha,
                      DeviceArray& a, int64_t lda,
                      DeviceArray& b, int64_t ldb,
                      DeviceArray& bias,
                      double beta,
                      DeviceArray& c, int64_t ldc) : params_{} {
    ROCBLAS_CALL_THROW(rocblas_create_handle(&rocblas_handle_));
    params_.tuning = true;
    params_.stream = Stream();
    params_.handle = rocblas_handle_;
    params_.opa = opa;
    params_.opb = opb;
    params_.m = m;
    params_.n = n;
    params_.k = k;
    params_.alpha = alpha;
    params_.a = static_cast<T*>(a.ptr());
    params_.lda = lda;
    params_.b = static_cast<T*>(b.ptr());
    params_.ldb = ldb;
    params_.bias = static_cast<T*>(bias.ptr());
    params_.beta = beta;
    params_.c = static_cast<T*>(c.ptr());
    params_.ldc = ldc;

    op_.EnableTuning();
  }

  ~GemmFastGeluTunable() {
    ROCBLAS_CALL_THROW(rocblas_destroy_handle(rocblas_handle_));
    rocblas_handle_ = nullptr;
  }

  void Run() override {
    ORT_THROW_IF_ERROR((op_(&params_)));
  }

  bool IsSupported() {
    Status status = op_(&params_);
    return status.IsOK();
  }

 private:
  using ParamsT = GemmFastGeluParams<T>;
  ParamsT params_{};
  rocblas_handle rocblas_handle_;
  GemmFastGeluTunableOp<T, ALayout, BLayout> op_{};
};

#define REGISTER_OP(name, type)                                \
  py::class_<name<type>>(m, #name "_" #type)                   \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, \
                    double,                                    \
                    DeviceArray&, int64_t,                     \
                    DeviceArray&, int64_t,                     \
                    DeviceArray&,                              \
                    double,                                    \
                    DeviceArray&, int64_t>())                  \
      .def("SetRepeats", &name<type>::SetRepeats)              \
      .def("Run", &name<type>::Run)                            \
      .def("Profile", &name<type>::Profile)                    \
      .def("IsSupported", &name<type>::IsSupported);

// #define REGISTER_OP_FOR_ALL_TRANSAB(type) \
//   REGISTER_OP(type, Row, Row, "NN");      \
//   REGISTER_OP(type, Row, Col, "NT");      \
//   REGISTER_OP(type, Col, Row, "TN");      \
//   REGISTER_OP(type, Col, Col, "TT");

void InitGemmFastGelu(py::module m) {
  REGISTER_OP(GemmFastGeluUnfused, float)
  REGISTER_OP(GemmFastGeluUnfused, half)

  // REGISTER_OP_FOR_ALL_TRANSAB(GemmFastGeluTunableOp, float)
  // REGISTER_OP_FOR_ALL_TRANSAB(GemmFastGeluTunableOp, half)
}

}  // namespace onnxruntime
