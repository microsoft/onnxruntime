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

using namespace rocm::tunable::blas::internal;

template <typename T, typename ALayout, typename BLayout>
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
    // rocblas handle is not used for hipBLASLt
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
    params_.bias = static_cast<T*>(bias.ptr());
    params_.beta = static_cast<float>(beta);
    params_.c = static_cast<T*>(c.ptr());
    params_.ldc = ldc;

    for (auto&& [type_string, op] : GetHipBlasLtGemmFastGeluTypeStringAndOps<T, ALayout, BLayout>()) {
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
  using ParamsT = contrib::rocm::blas::GemmFastGeluParams<T>;
  using OpT = Op<ParamsT>;
  ParamsT params_;
  std::vector<OpT> ops_;
  std::vector<std::string> type_strings_;
  size_t selected_op_{};
};

#define REGISTER_OP(type, alayout, blayout, layout_string)                                                       \
  py::class_<GemmFastGeluHipBlasLt<type, alayout, blayout>>(m, "GemmFastGeluHipBlasLt_" #type "_" layout_string) \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t,                                                   \
                    double,                                                                                      \
                    DeviceArray&, int64_t,                                                                       \
                    DeviceArray&, int64_t,                                                                       \
                    DeviceArray&,                                                                                \
                    double,                                                                                      \
                    DeviceArray&, int64_t>())                                                                    \
      .def("SetRepeats", &GemmFastGeluHipBlasLt<type, alayout, blayout>::SetRepeats)                             \
      .def("Profile", &GemmFastGeluHipBlasLt<type, alayout, blayout>::Profile)                                   \
      .def("Run", &GemmFastGeluHipBlasLt<type, alayout, blayout>::Run)                                           \
      .def("ListOps", &GemmFastGeluHipBlasLt<type, alayout, blayout>::ListOps)                                   \
      .def("SelectOp", &GemmFastGeluHipBlasLt<type, alayout, blayout>::SelectOp);

#define REGISTER_OP_FOR_ALL_TRANSAB(type) \
  REGISTER_OP(type, Row, Row, "NN");      \
  REGISTER_OP(type, Row, Col, "NT");      \
  REGISTER_OP(type, Col, Row, "TN");      \
  REGISTER_OP(type, Col, Col, "TT");

KE_REGISTER(m) {
  REGISTER_OP_FOR_ALL_TRANSAB(float);
  REGISTER_OP_FOR_ALL_TRANSAB(half);
}
#endif  // USE_HIPBLASLT

}  // namespace onnxruntime
