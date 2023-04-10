// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/gemm_tunable.cuh"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"
#include "python/tools/kernel_explorer/kernels/rocm/gemm_ke.h"

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
    params_.tuning_ctx = TuningContext();
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

    params_.TuningContext()->EnableTunableOpAndTuning();
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

template <typename T, typename ALayout, typename BLayout>
class BatchedGemmTunable : public IBatchedGemmKernelExplorer<T> {
 public:
  BatchedGemmTunable(BlasOp opa, BlasOp opb,
                     int64_t m, int64_t n, int64_t k,
                     double alpha,
                     std::vector<DeviceArray>& as, int64_t lda,
                     std::vector<DeviceArray>& bs, int64_t ldb,
                     double beta,
                     std::vector<DeviceArray>& cs, int64_t ldc,
                     int64_t batch) {
    this->CopyAsBsCsPointersToDevice(as, bs, cs, batch);

    ROCBLAS_CALL_THROW(rocblas_create_handle(&rocblas_handle_));
    params_.tuning_ctx = this->TuningContext();
    params_.stream = this->Stream();
    params_.handle = rocblas_handle_;
    params_.opa = opa;
    params_.opb = opb;
    params_.m = m;
    params_.n = n;
    params_.k = k;
    params_.alpha = alpha;
    params_.as = const_cast<const T**>(this->dev_as_.get());
    params_.lda = lda;
    params_.bs = const_cast<const T**>(this->dev_bs_.get());
    params_.ldb = ldb;
    params_.beta = beta;
    params_.cs = this->dev_cs_.get();
    params_.ldc = ldc;
    params_.batch = batch;

    params_.TuningContext()->EnableTunableOpAndTuning();
  }

  ~BatchedGemmTunable() {
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
  using ParamsT = BatchedGemmParams<T>;
  ParamsT params_;

  // tunable is stateful, store it as an instance
  BatchedGemmTunableOp<T, ALayout, BLayout> op_{};
  rocblas_handle rocblas_handle_;
};

template <typename T, typename ALayout, typename BLayout>
class StridedBatchedGemmTunable : public IKernelExplorer {
 public:
  StridedBatchedGemmTunable(BlasOp opa, BlasOp opb,
                            int64_t m, int64_t n, int64_t k,
                            double alpha,
                            DeviceArray& a, int64_t lda, int64_t stride_a,
                            DeviceArray& b, int64_t ldb, int64_t stride_b,
                            double beta,
                            DeviceArray& c, int64_t ldc, int64_t stride_c,
                            int64_t batch) {
    ROCBLAS_CALL_THROW(rocblas_create_handle(&rocblas_handle_));
    params_.tuning_ctx = TuningContext();
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
    params_.stride_a = stride_a;
    params_.b = static_cast<T*>(b.ptr());
    params_.ldb = ldb;
    params_.stride_b = stride_b;
    params_.beta = beta;
    params_.c = static_cast<T*>(c.ptr());
    params_.ldc = ldc;
    params_.stride_c = stride_c;
    params_.batch = batch;

    params_.TuningContext()->EnableTunableOpAndTuning();
  }

  ~StridedBatchedGemmTunable() {
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
  using ParamsT = StridedBatchedGemmParams<T>;
  ParamsT params_;

  // tunable is stateful, store it as an instance
  StridedBatchedGemmTunableOp<T, ALayout, BLayout> op_{};
  rocblas_handle rocblas_handle_;
};

#define REGISTER_OP_COMMON(type, dtype, alayout, blayout, layout_string)           \
  py::class_<type<dtype, alayout, blayout>>(m, #type "_" #dtype "_" layout_string) \
      .def("SetRepeats", &type<dtype, alayout, blayout>::SetRepeats)               \
      .def("Profile", &type<dtype, alayout, blayout>::Profile)                     \
      .def("Run", &type<dtype, alayout, blayout>::Run)                             \
      .def("ListOps", &type<dtype, alayout, blayout>::ListOps)                     \
      .def("SelectOp", &type<dtype, alayout, blayout>::SelectOp)

#define REGISTER_GEMM(dtype, alayout, blayout, layout_string)             \
  REGISTER_OP_COMMON(GemmTunable, dtype, alayout, blayout, layout_string) \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t,            \
                    double,                                               \
                    DeviceArray&, int64_t,                                \
                    DeviceArray&, int64_t,                                \
                    double,                                               \
                    DeviceArray&, int64_t>())

#define REGISTER_GEMM_FOR_ALL_TRANSAB(dtype) \
  REGISTER_GEMM(dtype, Row, Row, "NN");      \
  REGISTER_GEMM(dtype, Row, Col, "NT");      \
  REGISTER_GEMM(dtype, Col, Row, "TN");      \
  REGISTER_GEMM(dtype, Col, Col, "TT");

#define REGISTER_BATCHED_GEMM(dtype, alayout, blayout, layout_string)            \
  REGISTER_OP_COMMON(BatchedGemmTunable, dtype, alayout, blayout, layout_string) \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t,                   \
                    double,                                                      \
                    std::vector<DeviceArray>&, int64_t,                          \
                    std::vector<DeviceArray>&, int64_t,                          \
                    double,                                                      \
                    std::vector<DeviceArray>&, int64_t,                          \
                    int64_t>())

#define REGISTER_BATCHED_GEMM_FOR_ALL_TRANSAB(dtype) \
  REGISTER_BATCHED_GEMM(dtype, Row, Row, "NN");      \
  REGISTER_BATCHED_GEMM(dtype, Row, Col, "NT");      \
  REGISTER_BATCHED_GEMM(dtype, Col, Row, "TN");      \
  REGISTER_BATCHED_GEMM(dtype, Col, Col, "TT");

#define REGISTER_STRIDED_BATCHED_GEMM(dtype, alayout, blayout, layout_string)           \
  REGISTER_OP_COMMON(StridedBatchedGemmTunable, dtype, alayout, blayout, layout_string) \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t,                          \
                    double,                                                             \
                    DeviceArray&, int64_t, int64_t,                                     \
                    DeviceArray&, int64_t, int64_t,                                     \
                    double,                                                             \
                    DeviceArray&, int64_t, int64_t,                                     \
                    int64_t>())

#define REGISTER_STRIDED_BATCHED_GEMM_FOR_ALL_TRANSAB(dtype) \
  REGISTER_STRIDED_BATCHED_GEMM(dtype, Row, Row, "NN");      \
  REGISTER_STRIDED_BATCHED_GEMM(dtype, Row, Col, "NT");      \
  REGISTER_STRIDED_BATCHED_GEMM(dtype, Col, Row, "TN");      \
  REGISTER_STRIDED_BATCHED_GEMM(dtype, Col, Col, "TT");

KE_REGISTER(m) {
  REGISTER_GEMM_FOR_ALL_TRANSAB(float);
  REGISTER_GEMM_FOR_ALL_TRANSAB(half);

  REGISTER_BATCHED_GEMM_FOR_ALL_TRANSAB(float);
  REGISTER_BATCHED_GEMM_FOR_ALL_TRANSAB(half);

  REGISTER_STRIDED_BATCHED_GEMM_FOR_ALL_TRANSAB(float);
  REGISTER_STRIDED_BATCHED_GEMM_FOR_ALL_TRANSAB(half);
}

}  // namespace onnxruntime
