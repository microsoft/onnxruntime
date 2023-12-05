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
#include "contrib_ops/rocm/math/gemm_float8_ck.cuh"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

using namespace onnxruntime::rocm::tunable::blas;

namespace py = pybind11;

namespace onnxruntime {

#if defined(USE_COMPOSABLE_KERNEL) && !defined(DISABLE_FLOAT8_TYPES)
template <typename TA, typename TB, typename TC, typename ALayout, typename BLayout>
class GemmFloat8CK : public IKernelExplorer {
 public:
  GemmFloat8CK(BlasOp opa, BlasOp opb,
               int64_t m, int64_t n, int64_t k,
               float alpha,
               DeviceArray& a, int64_t lda, DeviceArray& scale_a,
               DeviceArray& b, int64_t ldb, DeviceArray& scale_b,
               float beta,
               DeviceArray& c, int64_t ldc, DeviceArray& scale_c)
      : params_{} {
    auto supports_a = opa == BlasOp::N ? std::is_same_v<ALayout, Row> : std::is_same_v<ALayout, Col>;
    auto supports_b = opb == BlasOp::N ? std::is_same_v<BLayout, Row> : std::is_same_v<BLayout, Col>;
    ORT_ENFORCE(supports_a && supports_b);

    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    // rocblas handle is not used for ck
    params_.handle = nullptr;
    params_.opa = opa;
    params_.opb = opb;
    params_.m = m;
    params_.n = n;
    params_.k = k;

    params_.a = static_cast<TA*>(a.ptr());
    params_.lda = lda;
    if constexpr (std::is_same_v<TA, Float8E4M3FN> || std::is_same_v<TA, Float8E4M3FNUZ>) {
      params_.scale_a = alpha;
      params_.scale_a_dev = static_cast<float*>(scale_a.ptr());
    }

    params_.b = static_cast<TB*>(b.ptr());
    params_.ldb = ldb;
    if constexpr (std::is_same_v<TB, Float8E4M3FN> || std::is_same_v<TB, Float8E4M3FNUZ>) {
      params_.scale_b = alpha;
      params_.scale_b_dev = static_cast<float*>(scale_b.ptr());
    }

    params_.c = static_cast<TC*>(c.ptr());
    params_.ldc = ldc;
    if constexpr (std::is_same_v<TC, Float8E4M3FN> || std::is_same_v<TC, Float8E4M3FNUZ>) {
      ORT_ENFORCE(false, "Not implemented");
      params_.scale_c = beta;
      params_.scale_c_dev = static_cast<float*>(scale_c.ptr());
    }

    for (auto&& [type_string, op] : GetCKF8SplitKGemmTypeStringAndOps<TA, TB, TC, ALayout, BLayout>()) {
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
  using ParamsT = GemmFloat8Params<TA, TB, TC>;
  using OpT = Op<ParamsT>;
  ParamsT params_{};
  std::vector<OpT> ops_;
  std::vector<std::string> type_strings_;
  size_t selected_op_{};
};

#define REGISTER_OP_COMMON(registered_name, tpl, dta, dtb, dtc, alayout, blayout) \
  py::class_<tpl<dta, dtb, dtc, alayout, blayout>>(m, registered_name)            \
      .def("SetRepeats", &tpl<dta, dtb, dtc, alayout, blayout>::SetRepeats)       \
      .def("Profile", &tpl<dta, dtb, dtc, alayout, blayout>::Profile)             \
      .def("Run", &tpl<dta, dtb, dtc, alayout, blayout>::Run)                     \
      .def("ListOps", &tpl<dta, dtb, dtc, alayout, blayout>::ListOps)             \
      .def("SelectOp", &tpl<dta, dtb, dtc, alayout, blayout>::SelectOp)

#define REGISTER_GEMM_FLOAT8_CK(registered_name, dta, dtb, dtc, alayout, blayout)    \
  REGISTER_OP_COMMON(registered_name, GemmFloat8CK, dta, dtb, dtc, alayout, blayout) \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t,                       \
                    float,                                                           \
                    DeviceArray&, int64_t, DeviceArray&,                             \
                    DeviceArray&, int64_t, DeviceArray&,                             \
                    float,                                                           \
                    DeviceArray&, int64_t, DeviceArray&>());

KE_REGISTER(m) {
  REGISTER_GEMM_FLOAT8_CK("GemmFloat8CK_fp8e4m3fn_half_half_NN", Float8E4M3FN, half, half, Row, Row);
  REGISTER_GEMM_FLOAT8_CK("GemmFloat8CK_half_fp8e4m3fn_half_NN", half, Float8E4M3FN, half, Row, Row);
  REGISTER_GEMM_FLOAT8_CK("GemmFloat8CK_fp8e4m3fnuz_half_half_NN", Float8E4M3FNUZ, half, half, Row, Row);
  REGISTER_GEMM_FLOAT8_CK("GemmFloat8CK_half_fp8e4m3fnuz_half_NN", half, Float8E4M3FNUZ, half, Row, Row);

  REGISTER_GEMM_FLOAT8_CK("GemmFloat8CK_half_fp8e4m3fn_half_NT", half, Float8E4M3FN, half, Row, Col);
  REGISTER_GEMM_FLOAT8_CK("GemmFloat8CK_half_fp8e4m3fnuz_half_NT", half, Float8E4M3FNUZ, half, Row, Col);
}
#endif

}  // namespace onnxruntime
