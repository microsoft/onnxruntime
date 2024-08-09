// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/gemm_ck.cuh"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

using namespace onnxruntime::rocm::tunable::blas;
using namespace onnxruntime::rocm::tunable::blas::internal;

namespace py = pybind11;

namespace onnxruntime {

#ifdef USE_COMPOSABLE_KERNEL
template <typename T, BlasOp OpA, BlasOp OpB>
class CKGemm : public IKernelExplorer {
 public:
  CKGemm(BlasOp opa, BlasOp opb,
         int64_t m, int64_t n, int64_t k,
         double alpha,
         DeviceArray& a, int64_t lda,
         DeviceArray& b, int64_t ldb,
         double beta,
         DeviceArray& c, int64_t ldc)
      : params_{} {
    ORT_ENFORCE(opa == OpA && opb == OpB);

    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    // rocblas handle is not used for ck
    params_.handle = nullptr;
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
    params_.beta = static_cast<float>(beta);
    params_.c = static_cast<T*>(c.ptr());
    params_.ldc = ldc;

    for (auto&& [type_string, op] : GetCKGemmTypeStringAndOps<T, OpA, OpB>()) {
      type_strings_.emplace_back(std::move(type_string));
      ops_.emplace_back(std::move(op));
    }
    for (auto&& [type_string, op] : GetCKStreamKGemmTypeStringAndOps<T, OpA, OpB>()) {
      type_strings_.emplace_back(std::move(type_string));
      ops_.emplace_back(std::move(op));
    }
    for (auto&& [type_string, op] : GetCKSplitKGemmTypeStringAndOps<T, OpA, OpB>()) {
      type_strings_.emplace_back(std::move(type_string));
      ops_.emplace_back(std::move(op));
    }
    ORT_ENFORCE(!ops_.empty());
  }

  void Run() override {
    ORT_THROW_IF_ERROR(ops_[selected_op_](&params_));
  }

  std::vector<std::string> ListOps() const {
    return type_strings_;
  }

  bool SelectOp(const std::string& name) {
    for (size_t i = 0; i < ops_.size(); i++) {
      if (type_strings_[i] == name) {
        selected_op_ = i;
        Status status = ops_[i](&params_);
        return status.IsOK();
      }
    }

    ORT_THROW("Cannot find implementation ", name);
  }

 private:
  using ParamsT = GemmParams<T>;
  using OpT = Op<ParamsT>;
  ParamsT params_;
  std::vector<OpT> ops_;
  std::vector<std::string> type_strings_;
  size_t selected_op_{};
};

template <typename T, BlasOp OpA, BlasOp OpB>
class CKStridedBatchedGemm : public IKernelExplorer {
 public:
  CKStridedBatchedGemm(
      BlasOp opa, BlasOp opb,
      int64_t m, int64_t n, int64_t k,
      double alpha,
      DeviceArray& a, int64_t lda, int64_t stride_a,
      DeviceArray& b, int64_t ldb, int64_t stride_b,
      double beta,
      DeviceArray& c, int64_t ldc, int64_t stride_c,
      int64_t batch)
      : params_{} {
    ORT_ENFORCE(opa == OpA && opb == OpB);

    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    // rocblas handle is not used for ck
    params_.handle = nullptr;
    params_.opa = opa;
    params_.opb = opb;
    params_.m = m;
    params_.n = n;
    params_.k = k;
    params_.alpha = static_cast<float>(alpha);
    params_.a = static_cast<T*>(a.ptr());
    params_.lda = lda;
    params_.stride_a = stride_a;
    params_.b = static_cast<T*>(b.ptr());
    params_.ldb = ldb;
    params_.stride_b = stride_b;
    params_.beta = static_cast<float>(beta);
    params_.c = static_cast<T*>(c.ptr());
    params_.ldc = ldc;
    params_.stride_c = stride_c;
    params_.batch = batch;

    for (auto&& [type_string, op] : GetCKStridedBatchedGemmTypeStringAndOps<T, OpA, OpB>()) {
      type_strings_.emplace_back(std::move(type_string));
      ops_.emplace_back(std::move(op));
    }
    ORT_ENFORCE(!ops_.empty());
  }

  void Run() override {
    ORT_THROW_IF_ERROR(ops_[selected_op_](&params_));
  }

  std::vector<std::string> ListOps() const {
    return type_strings_;
  }

  bool SelectOp(const std::string& name) {
    for (size_t i = 0; i < ops_.size(); i++) {
      if (type_strings_[i] == name) {
        selected_op_ = i;
        Status status = ops_[i](&params_);
        return status.IsOK();
      }
    }

    ORT_THROW("Cannot find implementation ", name);
  }

 private:
  using ParamsT = StridedBatchedGemmParams<T>;
  using OpT = Op<ParamsT>;
  ParamsT params_;
  std::vector<OpT> ops_;
  std::vector<std::string> type_strings_;
  size_t selected_op_{};
};

#define REGISTER_OP_COMMON(type, dtype, opa, opb, layout_string)           \
  py::class_<type<dtype, opa, opb>>(m, #type "_" #dtype "_" layout_string) \
      .def("SetRepeats", &type<dtype, opa, opb>::SetRepeats)               \
      .def("Profile", &type<dtype, opa, opb>::Profile)                     \
      .def("Run", &type<dtype, opa, opb>::Run)                             \
      .def("ListOps", &type<dtype, opa, opb>::ListOps)                     \
      .def("SelectOp", &type<dtype, opa, opb>::SelectOp)

#define REGISTER_CKGEMM(dtype, opa, opb, layout_string)        \
  REGISTER_OP_COMMON(CKGemm, dtype, opa, opb, layout_string)   \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, \
                    double,                                    \
                    DeviceArray&, int64_t,                     \
                    DeviceArray&, int64_t,                     \
                    double,                                    \
                    DeviceArray&, int64_t>());

#define REGISTER_CKGEMM_FOR_ALL_TRANSAB(dtype)        \
  REGISTER_CKGEMM(dtype, BlasOp::N, BlasOp::N, "NN"); \
  REGISTER_CKGEMM(dtype, BlasOp::N, BlasOp::T, "NT"); \
  REGISTER_CKGEMM(dtype, BlasOp::T, BlasOp::N, "TN"); \
  REGISTER_CKGEMM(dtype, BlasOp::T, BlasOp::T, "TT");

#define REGISTER_CKSTRIDEDBATCHEDGEMM(dtype, opa, opb, layout_string)      \
  REGISTER_OP_COMMON(CKStridedBatchedGemm, dtype, opa, opb, layout_string) \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t,             \
                    double,                                                \
                    DeviceArray&, int64_t, int64_t,                        \
                    DeviceArray&, int64_t, int64_t,                        \
                    double,                                                \
                    DeviceArray&, int64_t, int64_t,                        \
                    int64_t>());

#define REGISTER_CKSTRIDEDBATCHEDGEMM_FOR_ALL_TRANSAB(dtype)        \
  REGISTER_CKSTRIDEDBATCHEDGEMM(dtype, BlasOp::N, BlasOp::N, "NN"); \
  REGISTER_CKSTRIDEDBATCHEDGEMM(dtype, BlasOp::N, BlasOp::T, "NT"); \
  REGISTER_CKSTRIDEDBATCHEDGEMM(dtype, BlasOp::T, BlasOp::N, "TN"); \
  REGISTER_CKSTRIDEDBATCHEDGEMM(dtype, BlasOp::T, BlasOp::T, "TT");

KE_REGISTER(m) {
  REGISTER_CKGEMM_FOR_ALL_TRANSAB(float);
  REGISTER_CKGEMM_FOR_ALL_TRANSAB(half);

  REGISTER_CKSTRIDEDBATCHEDGEMM_FOR_ALL_TRANSAB(float);
  REGISTER_CKSTRIDEDBATCHEDGEMM_FOR_ALL_TRANSAB(half);
}
#endif  // USE_COMPOSABLE_KERNEL

}  // namespace onnxruntime
