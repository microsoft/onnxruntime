// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#ifdef USE_COMPOSABLE_KERNEL
#include "core/providers/rocm/composable_kernel_common.h"

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_add_fastgelu.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_fastgelu.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#endif

#include "contrib_ops/rocm/bert/gemm_fast_gelu_common.h"

using onnxruntime::rocm::ToHipType;

namespace onnxruntime {
namespace contrib {
namespace rocm {
namespace blas {
namespace internal {

#ifdef USE_COMPOSABLE_KERNEL

using onnxruntime::rocm::CKDataTypeAdaptor;
using onnxruntime::rocm::CKBlasOpAdaptor;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Nop = ck::tensor_operation::element_wise::PassThrough;
using AddFastGelu = ck::tensor_operation::element_wise::AddFastGelu;
using FastGelu = ck::tensor_operation::element_wise::FastGelu;

template <typename T, BlasOp OpA, BlasOp OpB>
auto GetCKGemmAddFastGeluTypeStringAndOps() {
  using CKDataType = typename CKDataTypeAdaptor<T>::type;
  using ALayout = typename CKBlasOpAdaptor<OpA>::type;
  using BLayout = typename CKBlasOpAdaptor<OpB>::type;
  using DeviceGemmAddFastGelu = ck::tensor_operation::device::DeviceGemmMultipleD<
      ALayout, BLayout, ck::Tuple<Row>, Row,
      CKDataType, CKDataType, ck::Tuple<CKDataType>, CKDataType,
      Nop, Nop, AddFastGelu>;
  using InstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemmAddFastGelu>;

  std::vector<std::pair<std::string, Op<GemmFastGeluParams<T>>>> ret;
  for (auto&& impl : InstanceFactory::GetInstances()) {
    auto type_string = onnxruntime::MakeString("withbias ", impl->GetTypeString());

    auto invoker = impl->MakeInvokerPointer();
    auto ck_gemmfastgelu_op = [impl = std::move(impl), invoker = std::move(invoker)](const GemmFastGeluParams<T>* params) -> Status {
      auto one = ToHipType<T>::FromFloat(1.0f);
      auto zero = ToHipType<T>::FromFloat(0.0f);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          params->alpha != one || params->beta != zero || params->bias == nullptr,
          impl->GetTypeString(), " only supports alpha == 1 and beta == 0 and bias != nullptr");

      auto nop = Nop{};
      auto addfastgelu = AddFastGelu{};
      auto arg = impl->MakeArgumentPointer(params->a, params->b, std::array<const void*, 1>{params->bias}, params->c,
                                           params->m, params->n, params->k,
                                           params->lda, params->ldb, std::array<ck::index_t, 1>{0}, params->ldc,
                                           nop, nop, addfastgelu);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                impl->GetTypeString(), " does not support the params");
      invoker->Run(arg.get(), StreamConfig{params->StreamHandle()});
      return Status::OK();
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemmfastgelu_op)));
  }
  return ret;
}

template <typename T, BlasOp OpA, BlasOp OpB>
auto GetCKGemmFastGeluTypeStringAndOps() {
  using CKDataType = typename CKDataTypeAdaptor<T>::type;
  using ALayout = typename CKBlasOpAdaptor<OpA>::type;
  using BLayout = typename CKBlasOpAdaptor<OpB>::type;
  using DeviceGemmFastGelu = ck::tensor_operation::device::DeviceGemmMultipleD<
      ALayout, BLayout, ck::Tuple<>, Row,
      CKDataType, CKDataType, ck::Tuple<>, CKDataType,
      Nop, Nop, FastGelu>;
  using InstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemmFastGelu>;

  std::vector<std::pair<std::string, Op<GemmFastGeluParams<T>>>> ret;
  for (auto&& impl : InstanceFactory::GetInstances()) {
    auto type_string = onnxruntime::MakeString("nobias ", impl->GetTypeString());
    auto invoker = impl->MakeInvokerPointer();
    auto ck_gemmfastgelu_op = [impl = std::move(impl), invoker = std::move(invoker)](const GemmFastGeluParams<T>* params) -> Status {
      auto one = ToHipType<T>::FromFloat(1.0f);
      auto zero = ToHipType<T>::FromFloat(0.0f);

      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          params->alpha != one || params->beta != zero || params->bias != nullptr,
          impl->GetTypeString(), " only supports alpha == 1 and beta == 0 and bias == nullptr");

      auto nop = Nop{};
      auto fastgelu = FastGelu{};
      auto arg = impl->MakeArgumentPointer(params->a, params->b,
                                           {},
                                           params->c,
                                           params->m, params->n, params->k,
                                           params->lda, params->ldb,
                                           {},
                                           params->ldc,
                                           nop, nop, fastgelu);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                impl->GetTypeString(), " does not support the params");
      invoker->Run(arg.get(), StreamConfig{params->StreamHandle()});
      return Status::OK();
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemmfastgelu_op)));
  }
  return ret;
}
#else
struct Row {};
struct Col {};
#endif  // USE_COMPOSABLE_KERNEL

}  // namespace internal
}  // namespace blas
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
