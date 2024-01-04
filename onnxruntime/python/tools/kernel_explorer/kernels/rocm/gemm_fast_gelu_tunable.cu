// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "contrib_ops/rocm/bert/gemm_fast_gelu_common.h"
#include "contrib_ops/rocm/bert/gemm_fast_gelu_tunable.cuh"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

using namespace onnxruntime::contrib::rocm::blas;
using namespace onnxruntime::contrib::rocm::blas::internal;

namespace py = pybind11;

namespace onnxruntime {
template <typename T, BlasOp OpA, BlasOp OpB>
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
    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    params_.handle = rocblas_handle_;
    params_.opa = opa;
    params_.opb = opb;
    params_.m = m;
    params_.n = n;
    params_.k = k;
    params_.alpha = static_cast<float>(alpha);
    params_.a = static_cast<T*>(a.ptr());
    params_.lda = lda;
    params_.b = static_cast<T*>(b.ptr());
    params_.ldb = ldb;
    params_.bias = static_cast<T*>(bias.ptr());
    params_.beta = static_cast<float>(beta);
    params_.c = static_cast<T*>(c.ptr());
    params_.ldc = ldc;

    params_.TuningContext()->EnableTunableOpAndTuning();
  }

  ~GemmFastGeluTunable() {
    ROCBLAS_CALL_THROW(rocblas_destroy_handle(rocblas_handle_));
    rocblas_handle_ = nullptr;
  }

  void Run() override {
    WithMaxTuningDurationMs max_duration(TuningContext(), 250);
    ORT_THROW_IF_ERROR((op_(&params_)));
  }

  std::vector<std::string> ListOps() const {
    return {"GemmFastGeluTunable"};
  }

  bool SelectOp(const std::string& name) {
    return name == "GemmFastGeluTunable";
  }

 private:
  using ParamsT = GemmFastGeluParams<T>;
  ParamsT params_{};
  rocblas_handle rocblas_handle_;
  GemmFastGeluTunableOp<T, OpA, OpB> op_{};
};

#define REGISTER_OP(type, opa, opb, layout_string)                                                   \
  py::class_<GemmFastGeluTunable<type, opa, opb>>(m, "GemmFastGeluTunable_" #type "_" layout_string) \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t,                                       \
                    double,                                                                          \
                    DeviceArray&, int64_t,                                                           \
                    DeviceArray&, int64_t,                                                           \
                    DeviceArray&,                                                                    \
                    double,                                                                          \
                    DeviceArray&, int64_t>())                                                        \
      .def("SetRepeats", &GemmFastGeluTunable<type, opa, opb>::SetRepeats)                           \
      .def("Profile", &GemmFastGeluTunable<type, opa, opb>::Profile)                                 \
      .def("Run", &GemmFastGeluTunable<type, opa, opb>::Run)                                         \
      .def("ListOps", &GemmFastGeluTunable<type, opa, opb>::ListOps)                                 \
      .def("SelectOp", &GemmFastGeluTunable<type, opa, opb>::SelectOp);

#define REGISTER_OP_FOR_ALL_TRANSAB(type)        \
  REGISTER_OP(type, BlasOp::N, BlasOp::N, "NN"); \
  REGISTER_OP(type, BlasOp::N, BlasOp::T, "NT"); \
  REGISTER_OP(type, BlasOp::T, BlasOp::N, "TN"); \
  REGISTER_OP(type, BlasOp::T, BlasOp::T, "TT");

KE_REGISTER(m) {
  REGISTER_OP_FOR_ALL_TRANSAB(float);
  REGISTER_OP_FOR_ALL_TRANSAB(half);
}

}  // namespace onnxruntime
