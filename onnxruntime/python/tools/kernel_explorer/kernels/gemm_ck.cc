// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/gemm_ck.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "python/tools/kernel_explorer/kernels/gemm.h"

namespace py = pybind11;

namespace onnxruntime {

namespace {

template <typename T>
struct DataTypeAdaptor {
  using type = T;
};

template <>
struct DataTypeAdaptor<half> {
  using type = ck::half_t;
};

}  // namespace

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Nop = ck::tensor_operation::element_wise::PassThrough;

// to be moved to onnxruntime once we have a monolithicly tunable gemm wrapper and it is enabled for onnxruntime
template <typename T, typename ALayout, typename BLayout>
auto GetCKGemmTypeStringAndOps() {
  using CKDataType = typename DataTypeAdaptor<T>::type;
  using DeviceGemm = ck::tensor_operation::device::DeviceGemm<
      ALayout, BLayout, Row,
      CKDataType, CKDataType, CKDataType,
      Nop, Nop, Nop>;
  using InstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemm>;

  std::vector<std::pair<std::string, contrib::rocm::Op<GemmParams<T>>>> ret;
  for (auto&& impl : InstanceFactory::GetInstances()) {
    auto type_string = impl->GetTypeString();
    auto invoker = impl->MakeInvokerPointer();
    auto ck_gemm_op = [impl = std::move(impl), invoker = std::move(invoker)](const GemmParams<T>* params) -> Status {
      auto nop = Nop{};
      auto arg = impl->MakeArgumentPointer(params->a, params->b, params->c,
                                           params->m, params->n, params->k,
                                           params->lda, params->ldb, params->ldc,
                                           nop, nop, nop);
      TUNABLE_OP_RETURN_UNSUPPOTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                               impl->GetTypeString(), " does not support ", params->Signature());
      invoker->Run(arg.get(), StreamConfig{params->stream});
      return Status::OK();
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemm_op)));
  }
  return ret;
}

template <typename T, typename ALayout, typename BLayout>
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
    auto supports_a = opa == BlasOp::N ? std::is_same_v<ALayout, Row> : std::is_same_v<ALayout, Col>;
    auto supports_b = opb == BlasOp::N ? std::is_same_v<BLayout, Row> : std::is_same_v<BLayout, Col>;
    ORT_ENFORCE(supports_a && supports_b);

    // rocblas handle is not used for ck
    params_.handle = nullptr;
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

    for (auto&& [type_string, op] : GetCKGemmTypeStringAndOps<T, ALayout, BLayout>()) {
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
  using OpT = contrib::rocm::Op<ParamsT>;
  ParamsT params_;
  std::vector<OpT> ops_;
  std::vector<std::string> type_strings_;
  size_t selected_op_{};
};

#define REGISTER_OP(type, alayout, blayout, layout_string)                         \
  py::class_<CKGemm<type, alayout, blayout>>(m, "CKGemm_" #type "_" layout_string) \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t,                     \
                    double,                                                        \
                    DeviceArray&, int64_t,                                         \
                    DeviceArray&, int64_t,                                         \
                    double,                                                        \
                    DeviceArray&, int64_t>())                                      \
      .def("SetRepeats", &CKGemm<type, alayout, blayout>::SetRepeats)              \
      .def("Profile", &CKGemm<type, alayout, blayout>::Profile)                    \
      .def("Run", &CKGemm<type, alayout, blayout>::Run)                            \
      .def("ListOps", &CKGemm<type, alayout, blayout>::ListOps)                    \
      .def("SelectOp", &CKGemm<type, alayout, blayout>::SelectOp);

#define REGISTER_OP_FOR_ALL_TRANSAB(type) \
  REGISTER_OP(type, Row, Row, "NN");      \
  REGISTER_OP(type, Row, Col, "NT");      \
  REGISTER_OP(type, Col, Row, "TN");      \
  REGISTER_OP(type, Col, Col, "TT");

void InitComposableKernelGemm(py::module m) {
  REGISTER_OP_FOR_ALL_TRANSAB(float);
  REGISTER_OP_FOR_ALL_TRANSAB(half);
}

}  // namespace onnxruntime
