// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/gemm_rocblas.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/gemm_rocblas.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

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

void InitRocBlasGemm(py::module mod) {
  // float
  py::class_<RocBlasGemm<float>>(mod, "RocblasGemm_float")
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, double,
                    DeviceArray&, int64_t, DeviceArray&, int64_t, double, DeviceArray&, int64_t>())
      .def("SetRepeats", &RocBlasGemm<float>::SetRepeats)
      .def("Profile", &RocBlasGemm<float>::Profile)
      .def("Run", &RocBlasGemm<float>::Run)
      .def("ListOps", &RocBlasGemm<float>::ListOps)
      .def("SelectOp", &RocBlasGemm<float>::SelectOp);

  // half
  py::class_<RocBlasGemm<half>>(mod, "RocblasGemm_half")
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, double,
                    DeviceArray&, int64_t, DeviceArray&, int64_t, double, DeviceArray&, int64_t>())
      .def("SetRepeats", &RocBlasGemm<half>::SetRepeats)
      .def("Profile", &RocBlasGemm<half>::Profile)
      .def("Run", &RocBlasGemm<half>::Run)
      .def("ListOps", &RocBlasGemm<half>::ListOps)
      .def("SelectOp", &RocBlasGemm<half>::SelectOp);
}

}  // namespace onnxruntime
