// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/stl.h>

#include <string>
#include <vector>

#ifdef USE_HIPBLASLT
#include "core/providers/rocm/tunable/gemm_hipblaslt.h"
#endif

#include "contrib_ops/rocm/bert/gemm_fast_gelu_common.h"
#include "core/providers/rocm/rocm_common.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

namespace py = pybind11;

namespace onnxruntime {

#ifdef USE_HIPBLASLT
template <typename T>
class GemmFastGeluHipBlasLt : public IKernelExplorer {
 public:
  GemmFastGeluHipBlasLt(BlasOp opa, BlasOp opb,
                        int64_t m, int64_t n, int64_t k,
                        double alpha,
                        DeviceArray& a, int64_t lda,
                        DeviceArray& b, int64_t ldb,
                        DeviceArray& bias,
                        double beta,
                        DeviceArray& c, int64_t ldc) : params_{} {
    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
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

  void Run() override {
    ORT_THROW_IF_ERROR((rocm::tunable::blas::internal::HipBlasLtGemmFastGeluOp<T>(&params_)));
  }

  std::vector<std::string> ListOps() const {
    return {"GemmFastGeluHipBlasLt"};
  }

  bool SelectOp(const std::string& name) {
    Status status = rocm::tunable::blas::internal::HipBlasLtGemmFastGeluOp<T>(&params_);
    return status.IsOK() && name == "GemmFastGeluHipBlasLt";
  }

 private:
  using ParamsT = contrib::rocm::blas::GemmFastGeluParams<T>;
  ParamsT params_{};
};

#define REGISTER_OP(type)                                                    \
  py::class_<GemmFastGeluHipBlasLt<type>>(m, "GemmFastGeluHipBlasLt_" #type) \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t,               \
                    double,                                                  \
                    DeviceArray&, int64_t,                                   \
                    DeviceArray&, int64_t,                                   \
                    DeviceArray&,                                            \
                    double,                                                  \
                    DeviceArray&, int64_t>())                                \
      .def("SetRepeats", &GemmFastGeluHipBlasLt<type>::SetRepeats)           \
      .def("Run", &GemmFastGeluHipBlasLt<type>::Run)                         \
      .def("Profile", &GemmFastGeluHipBlasLt<type>::Profile)                 \
      .def("ListOps", &GemmFastGeluHipBlasLt<type>::ListOps)                 \
      .def("SelectOp", &GemmFastGeluHipBlasLt<type>::SelectOp);

KE_REGISTER(m) {
  REGISTER_OP(float)
  REGISTER_OP(half)
}
#endif  // USE_HIPBLASLT

}  // namespace onnxruntime
