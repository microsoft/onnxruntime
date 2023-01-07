// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/rocm/gemm_rocblas.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/gemm_rocblas.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"
#include "python/tools/kernel_explorer/kernels/rocm/gemm_ke.h"

using namespace onnxruntime::rocm::tunable::blas;
using namespace onnxruntime::rocm::tunable::blas::internal;

namespace py = pybind11;

namespace onnxruntime {

template <typename T>
class RocBlasGemm : public IKernelExplorer {
 public:
  RocBlasGemm(BlasOp opa, BlasOp opb,
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
  }

  ~RocBlasGemm() {
    ROCBLAS_CALL_THROW(rocblas_destroy_handle(rocblas_handle_));
    rocblas_handle_ = nullptr;
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  std::vector<std::string> ListOps() const {
    return {"Rocblas"};
  }

  bool SelectOp(const std::string& name) {
    return name == "Rocblas";
  }

 private:
  rocblas_handle rocblas_handle_;

  using ParamsT = GemmParams<T>;
  using OpT = rocm::tunable::Op<ParamsT>;

  ParamsT params_{};
  OpT op_{RocBlasGemmOp<T>};
};

template <typename T>
class RocBlasBatchedGemm : public IBatchedGemmKernelExplorer<T> {
 public:
  RocBlasBatchedGemm(BlasOp opa, BlasOp opb,
                     int64_t m, int64_t n, int64_t k,
                     double alpha,
                     std::vector<DeviceArray>& as, int64_t lda,
                     std::vector<DeviceArray>& bs, int64_t ldb,
                     double beta,
                     std::vector<DeviceArray>& cs, int64_t ldc,
                     int64_t batch) {
    this->CopyAsBsCsPointersToDevice(as, bs, cs, batch);
    ROCBLAS_CALL_THROW(rocblas_create_handle(&rocblas_handle_));
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
  }

  ~RocBlasBatchedGemm() {
    ROCBLAS_CALL_THROW(rocblas_destroy_handle(rocblas_handle_));
    rocblas_handle_ = nullptr;
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  std::vector<std::string> ListOps() const {
    return {"Rocblas"};
  }

  bool SelectOp(const std::string& name) {
    return name == "Rocblas";
  }

 private:
  rocblas_handle rocblas_handle_;

  using ParamsT = BatchedGemmParams<T>;
  using OpT = rocm::tunable::Op<ParamsT>;

  ParamsT params_{};
  OpT op_{RocBlasBatchedGemmOp<T>};
};

template <typename T>
class RocBlasStridedBatchedGemm : public IKernelExplorer {
 public:
  RocBlasStridedBatchedGemm(BlasOp opa, BlasOp opb,
                            int64_t m, int64_t n, int64_t k,
                            double alpha,
                            DeviceArray& a, int64_t lda, int64_t stride_a,
                            DeviceArray& b, int64_t ldb, int64_t stride_b,
                            double beta,
                            DeviceArray& c, int64_t ldc, int64_t stride_c,
                            int64_t batch) {
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
    params_.stride_a = stride_a;
    params_.b = static_cast<T*>(b.ptr());
    params_.ldb = ldb;
    params_.stride_b = stride_b;
    params_.beta = beta;
    params_.c = static_cast<T*>(c.ptr());
    params_.ldc = ldc;
    params_.stride_c = stride_c;
    params_.batch = batch;
  }

  ~RocBlasStridedBatchedGemm() {
    ROCBLAS_CALL_THROW(rocblas_destroy_handle(rocblas_handle_));
    rocblas_handle_ = nullptr;
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  std::vector<std::string> ListOps() const {
    return {"Rocblas"};
  }

  bool SelectOp(const std::string& name) {
    return name == "Rocblas";
  }

 private:
  rocblas_handle rocblas_handle_;

  using ParamsT = StridedBatchedGemmParams<T>;
  using OpT = rocm::tunable::Op<ParamsT>;

  ParamsT params_{};
  OpT op_{RocBlasStridedBatchedGemmOp<T>};
};

#define REGISTER_OP_COMMON(type, dtype)            \
  py::class_<type<dtype>>(mod, #type "_" #dtype)   \
      .def("SetRepeats", &type<dtype>::SetRepeats) \
      .def("Profile", &type<dtype>::Profile)       \
      .def("Run", &type<dtype>::Run)               \
      .def("ListOps", &type<dtype>::ListOps)       \
      .def("SelectOp", &type<dtype>::SelectOp)

#define REGISTER_GEMM(dtype)                                   \
  REGISTER_OP_COMMON(RocBlasGemm, dtype)                       \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, \
                    double,                                    \
                    DeviceArray&, int64_t,                     \
                    DeviceArray&, int64_t,                     \
                    double,                                    \
                    DeviceArray&, int64_t>())

#define REGISTER_BATCHED_GEMM(dtype)                           \
  REGISTER_OP_COMMON(RocBlasBatchedGemm, dtype)                \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, \
                    double,                                    \
                    std::vector<DeviceArray>&, int64_t,        \
                    std::vector<DeviceArray>&, int64_t,        \
                    double,                                    \
                    std::vector<DeviceArray>&, int64_t,        \
                    int64_t>())

#define REGISTER_STRIDED_BATCHED_GEMM(dtype)                   \
  REGISTER_OP_COMMON(RocBlasStridedBatchedGemm, dtype)         \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, \
                    double,                                    \
                    DeviceArray&, int64_t, int64_t,            \
                    DeviceArray&, int64_t, int64_t,            \
                    double,                                    \
                    DeviceArray&, int64_t, int64_t,            \
                    int64_t>())

void InitRocBlasGemm(py::module mod) {
  REGISTER_GEMM(float);
  REGISTER_GEMM(half);

  REGISTER_BATCHED_GEMM(float);
  REGISTER_BATCHED_GEMM(half);

  REGISTER_STRIDED_BATCHED_GEMM(float);
  REGISTER_STRIDED_BATCHED_GEMM(half);
}

}  // namespace onnxruntime
