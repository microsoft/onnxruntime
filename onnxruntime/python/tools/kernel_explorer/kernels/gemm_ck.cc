// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/gemm_ck.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"

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

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

}  // namespace

template <typename T>
class CKGemm : public GemmBase<T> {
  using CKDataType = typename DataTypeAdaptor<T>::type;

  template <typename ALayout, typename BLayout>
  using DeviceGemm = ck::tensor_operation::device::DeviceGemm<
      ALayout, BLayout, Row,
      CKDataType, CKDataType, CKDataType,
      ck::tensor_operation::element_wise::PassThrough,
      ck::tensor_operation::element_wise::PassThrough,
      ck::tensor_operation::element_wise::PassThrough>;

  template <typename Op>
  using CKGemmInstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<Op>;

  // composable kernel's GetInstances() returns a collection of <std::unique_ptr<DeviceGemm<...>>>. We need to manually
  // cast them to std::unique_ptr<BaseOperator>> to avoid type problem.
  template <typename Op>
  static std::vector<std::unique_ptr<ck::tensor_operation::device::BaseOperator>> GetInstances() {
    auto ptrs = CKGemmInstanceFactory<Op>::GetInstances();
    std::vector<std::unique_ptr<ck::tensor_operation::device::BaseOperator>> results;
    for (auto&& ptr : ptrs) {
      results.emplace_back(std::move(ptr));
    }
    return results;
  }

  bool UpdateArgumentAndInvoker() {
    auto nop = ck::tensor_operation::element_wise::PassThrough{};
    if (this->opa_ == BlasOp::N && this->opb_ == BlasOp::N) {
      using DeviceOp = DeviceGemm<Row, Row>;
      auto typed_impl = static_cast<DeviceOp*>(impls_[selected_impl_].get());
      arg_ = typed_impl->MakeArgumentPointer(this->a_, this->b_, this->c_,
                                             this->m_, this->n_, this->k_,
                                             this->lda_, this->ldb_, this->ldc_,
                                             nop, nop, nop);
      invoker_ = typed_impl->MakeInvokerPointer();
    } else if (this->opa_ == BlasOp::T && this->opb_ == BlasOp::N) {
      using DeviceOp = DeviceGemm<Col, Row>;
      auto typed_impl = static_cast<DeviceOp*>(impls_[selected_impl_].get());
      arg_ = typed_impl->MakeArgumentPointer(this->a_, this->b_, this->c_,
                                             this->m_, this->n_, this->k_,
                                             this->lda_, this->ldb_, this->ldc_,
                                             nop, nop, nop);
      invoker_ = typed_impl->MakeInvokerPointer();
    } else if (this->opa_ == BlasOp::N && this->opb_ == BlasOp::T) {
      using DeviceOp = DeviceGemm<Row, Col>;
      auto typed_impl = static_cast<DeviceOp*>(impls_[selected_impl_].get());
      arg_ = typed_impl->MakeArgumentPointer(this->a_, this->b_, this->c_,
                                             this->m_, this->n_, this->k_,
                                             this->lda_, this->ldb_, this->ldc_,
                                             nop, nop, nop);
      invoker_ = typed_impl->MakeInvokerPointer();
    } else if (this->opa_ == BlasOp::T && this->opb_ == BlasOp::T) {
      using DeviceOp = DeviceGemm<Col, Col>;
      auto typed_impl = static_cast<DeviceOp*>(impls_[selected_impl_].get());
      arg_ = typed_impl->MakeArgumentPointer(this->a_, this->b_, this->c_,
                                             this->m_, this->n_, this->k_,
                                             this->lda_, this->ldb_, this->ldc_,
                                             nop, nop, nop);
      invoker_ = typed_impl->MakeInvokerPointer();
    }

    return impls_[selected_impl_]->IsSupportedArgument(arg_.get());
  }

 public:
  CKGemm(BlasOp opa, BlasOp opb,
         int64_t m, int64_t n, int64_t k,
         double alpha,
         DeviceArray& a, int64_t lda,
         DeviceArray& b, int64_t ldb,
         double beta,
         DeviceArray& c, int64_t ldc)
      : GemmBase<T>(opa, opb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) {
    ORT_ENFORCE(alpha == 1.0);
    ORT_ENFORCE(beta == 0.0);

    if (opa == BlasOp::N && opb == BlasOp::N) {
      impls_ = this->GetInstances<DeviceGemm<Row, Row>>();
    } else if (opa == BlasOp::T && opb == BlasOp::N) {
      impls_ = this->GetInstances<DeviceGemm<Col, Row>>();
    } else if (opa == BlasOp::N && opb == BlasOp::T) {
      impls_ = this->GetInstances<DeviceGemm<Row, Col>>();
    } else if (opa == BlasOp::T && opb == BlasOp::T) {
      impls_ = this->GetInstances<DeviceGemm<Col, Col>>();
    }

    ORT_ENFORCE(!impls_.empty());
    selected_impl_ = 0;
    UpdateArgumentAndInvoker();
  }

  std::vector<std::string> ListImpls() const override {
    std::vector<std::string> results;
    std::transform(impls_.cbegin(), impls_.cend(), std::back_inserter(results),
                   [](const auto& it) { return it->GetTypeString(); });
    return results;
  }

  bool SelectImpl(const std::string& name) override {
    for (size_t i = 0; i < impls_.size(); i++) {
      if (impls_[i]->GetTypeString() == name) {
        selected_impl_ = i;
        return UpdateArgumentAndInvoker();
      }
    }

    ORT_THROW("Cannot find implementation ", name);
  }

  void Run() override {
    invoker_->Run(arg_.get());
  }

 private:
  std::vector<std::unique_ptr<ck::tensor_operation::device::BaseOperator>> impls_;
  size_t selected_impl_{};
  std::unique_ptr<ck::tensor_operation::device::BaseArgument> arg_;
  std::unique_ptr<ck::tensor_operation::device::BaseInvoker> invoker_;
};

void InitComposableKernelGemm(py::module mod) {
  // float
  py::class_<CKGemm<float>>(mod, "CKGemm_float")
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, double,
                    DeviceArray&, int64_t, DeviceArray&, int64_t, double, DeviceArray&, int64_t>())
      .def("SetRepeats", &CKGemm<float>::SetRepeats)
      .def("Profile", &CKGemm<float>::Profile)
      .def("Run", &CKGemm<float>::Run)
      .def("ListImpls", &CKGemm<float>::ListImpls)
      .def("SelectImpl", &CKGemm<float>::SelectImpl);

  // half
  py::class_<CKGemm<half>>(mod, "CKGemm_half")
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t, double,
                    DeviceArray&, int64_t, DeviceArray&, int64_t, double, DeviceArray&, int64_t>())
      .def("SetRepeats", &CKGemm<half>::SetRepeats)
      .def("Profile", &CKGemm<half>::Profile)
      .def("Run", &CKGemm<half>::Run)
      .def("ListImpls", &CKGemm<half>::ListImpls)
      .def("SelectImpl", &CKGemm<half>::SelectImpl);
}

}  // namespace onnxruntime
