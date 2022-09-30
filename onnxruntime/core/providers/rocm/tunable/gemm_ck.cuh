// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "core/providers/rocm/tunable/gemm_common.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

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

using Nop = ck::tensor_operation::element_wise::PassThrough;

template <typename T, typename ALayout, typename BLayout>
auto GetCKGemmTypeStringAndOps() {
  using CKDataType = typename DataTypeAdaptor<T>::type;
  using DeviceGemm = ck::tensor_operation::device::DeviceGemm<
      ALayout, BLayout, Row,
      CKDataType, CKDataType, CKDataType,
      Nop, Nop, Nop>;
  using InstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemm>;

  std::vector<std::pair<std::string, Op<GemmParams<T>>>> ret;
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

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
