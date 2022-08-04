// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/gemm_rocblas.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernels/gemm.h"

namespace py = pybind11;

namespace onnxruntime {

template <typename T>
class RocBlasGemm : public GemmBase<T> {
 public:
  RocBlasGemm(BlasOp opa, BlasOp opb,
              int64_t m, int64_t n, int64_t k,
              double alpha,
              DeviceArray& a, int64_t lda,
              DeviceArray& b, int64_t ldb,
              double beta,
              DeviceArray& c, int64_t ldc)
      : GemmBase<T>(opa, opb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        opa_(opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose),
        opb_(opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose) {
    ROCBLAS_CALL_THROW(rocblas_create_handle(&rocblas_handle_));
  }

  ~RocBlasGemm() {
    ROCBLAS_CALL_THROW(rocblas_destroy_handle(rocblas_handle_));
    rocblas_handle_ = nullptr;
  }

  void Run() override {
    // NOTE: rocblas assumes the storage is column-majored, swapping A and B makes it have the same interface
    // as those with row-majored convention. That is, if you treat the storage as row-majored but view the matrices as
    // transposed, then by using the property Transpose(A*B) = Tranpose(B)*Transpose(A), the correctness is obvious.
    ROCBLAS_CALL_THROW(
        rocblasGemmHelper(this->rocblas_handle_, this->opb_, this->opa_,
                          this->n_, this->m_, this->k_,
                          &(this->alpha_),
                          this->b_, this->ldb_,
                          this->a_, this->lda_,
                          &(this->beta_),
                          this->c_, this->ldc_));
  }

  std::vector<std::string> ListImpls() const override {
    return {"Rocblas"};
  }

  bool SelectImpl(const std::string& name) override {
    return name == "Rocblas";
  }

 private:
  rocblas_handle rocblas_handle_;
  rocblas_operation opa_;
  rocblas_operation opb_;
};

void InitRocBlasGemm(py::module mod) {
  // float
  py::class_<RocBlasGemm<float>>(mod, "RocblasGemm_float")
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, double,
                    DeviceArray&, int64_t, DeviceArray&, int64_t, double, DeviceArray&, int64_t>())
      .def("SetRepeats", &RocBlasGemm<float>::SetRepeats)
      .def("Profile", &RocBlasGemm<float>::Profile)
      .def("Run", &RocBlasGemm<float>::Run)
      .def("ListImpls", &RocBlasGemm<float>::ListImpls)
      .def("SelectImpl", &RocBlasGemm<float>::SelectImpl);

  // half
  py::class_<RocBlasGemm<half>>(mod, "RocblasGemm_half")
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, double,
                    DeviceArray&, int64_t, DeviceArray&, int64_t, double, DeviceArray&, int64_t>())
      .def("SetRepeats", &RocBlasGemm<half>::SetRepeats)
      .def("Profile", &RocBlasGemm<half>::Profile)
      .def("Run", &RocBlasGemm<half>::Run)
      .def("ListImpls", &RocBlasGemm<half>::ListImpls)
      .def("SelectImpl", &RocBlasGemm<half>::SelectImpl);
}

}  // namespace onnxruntime
