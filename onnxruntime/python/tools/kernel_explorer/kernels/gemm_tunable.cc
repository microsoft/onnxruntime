// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/gemm_tunable.h"

#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/gemm_tunable.cuh"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

using namespace onnxruntime::rocm::tunable::blas;
using namespace onnxruntime::rocm::tunable::blas::internal;

namespace onnxruntime {

template <typename T, typename ALayout, typename BLayout>
class GemmTunable : public IKernelExplorer {
 public:
  GemmTunable(BlasOp opa, BlasOp opb,
              int64_t m, int64_t n, int64_t k,
              double alpha,
              DeviceArray& a, int64_t lda,
              DeviceArray& b, int64_t ldb,
              double beta,
              DeviceArray& c, int64_t ldc) {
    ROCBLAS_CALL_THROW(rocblas_create_handle(&rocblas_handle_));
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
    params_.beta = beta;
    params_.c = static_cast<T*>(c.ptr());
    params_.ldc = ldc;

    op_.EnableTuning();
  }

  ~GemmTunable() {
    ROCBLAS_CALL_THROW(rocblas_destroy_handle(rocblas_handle_));
    rocblas_handle_ = nullptr;
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  std::vector<std::string> ListOps() const {
    return {"Tunable"};
  }

  bool SelectOp(const std::string& name) {
    return name == "Tunable";
  }

 private:
  using ParamsT = GemmParams<T>;
  ParamsT params_;

  // tunable is stateful, store it as an instance
  GemmTunableOp<T, ALayout, BLayout> op_{};
  rocblas_handle rocblas_handle_;
};

#define REGISTER_OP(type, alayout, blayout, layout_string)                                   \
  py::class_<GemmTunable<type, alayout, blayout>>(m, "GemmTunable_" #type "_" layout_string) \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t,                               \
                    double,                                                                  \
                    DeviceArray&, int64_t,                                                   \
                    DeviceArray&, int64_t,                                                   \
                    double,                                                                  \
                    DeviceArray&, int64_t>())                                                \
      .def("SetRepeats", &GemmTunable<type, alayout, blayout>::SetRepeats)                   \
      .def("Profile", &GemmTunable<type, alayout, blayout>::Profile)                         \
      .def("Run", &GemmTunable<type, alayout, blayout>::Run)                                 \
      .def("ListOps", &GemmTunable<type, alayout, blayout>::ListOps)                         \
      .def("SelectOp", &GemmTunable<type, alayout, blayout>::SelectOp);

#define REGISTER_OP_FOR_ALL_TRANSAB(type) \
  REGISTER_OP(type, Row, Row, "NN");      \
  REGISTER_OP(type, Row, Col, "NT");      \
  REGISTER_OP(type, Col, Row, "TN");      \
  REGISTER_OP(type, Col, Col, "TT");

void InitTunableGemm(py::module m) {
  REGISTER_OP_FOR_ALL_TRANSAB(float);
  REGISTER_OP_FOR_ALL_TRANSAB(half);
}

}  // namespace onnxruntime
