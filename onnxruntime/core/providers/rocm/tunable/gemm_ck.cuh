// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#ifdef USE_COMPOSABLE_KERNEL
#include "core/providers/rocm/composable_kernel_common.h"

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#endif

#include "core/providers/rocm/tunable/gemm_common.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

#ifdef USE_COMPOSABLE_KERNEL

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Nop = ck::tensor_operation::element_wise::PassThrough;

template <typename T, typename ALayout, typename BLayout>
auto GetCKGemmTypeStringAndOps() {
  using CKDataType = typename CKDataTypeAdaptor<T>::type;
  using DeviceGemm = ck::tensor_operation::device::DeviceGemm<
      ALayout, BLayout, Row,
      CKDataType, CKDataType, CKDataType,
      Nop, Nop, Nop>;
  using InstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemm>;

  std::vector<std::pair<std::string, Op<GemmParams<T>>>> ret;
  for (auto&& impl : InstanceFactory::GetInstances()) {
    auto type_string = impl->GetTypeString();

    // FIXME: ck upstream have bugs in some input shapes coupled with specific impls. The `IsSupportedArgument` is not
    // sound, we exclude those implementation here for now. Check back later when AMD fixed them.
    //
    // The DeviceGemmXdl<256, 128, 144, 8, 8, 16, 16, 2, 9> and DeviceGemmXdl<256, 128, 144, 4, 8, 16, 16, 2, 9> only
    // occurs in DeviceGemm<Row, Col> for FP16. When k < 8, the result is wrong.
    if (type_string == "DeviceGemmXdl<256, 128, 144, 8, 8, 16, 16, 2, 9>" ||
        type_string == "DeviceGemmXdl<256, 128, 144, 4, 8, 16, 16, 2, 9>") {
      continue;
    }

    auto invoker = impl->MakeInvokerPointer();
    auto ck_gemm_op = [impl = std::move(impl), invoker = std::move(invoker)](const GemmParams<T>* params) -> Status {
      auto one = ToHipType<T>::FromFloat(1.0f);
      auto zero = ToHipType<T>::FromFloat(0.0f);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          params->alpha != one || params->beta != zero,
          impl->GetTypeString(), " only supports alpha == 1 and beta == 0", params->Signature());

      auto nop = Nop{};
      auto arg = impl->MakeArgumentPointer(params->a, params->b, params->c,
                                           params->m, params->n, params->k,
                                           params->lda, params->ldb, params->ldc,
                                           nop, nop, nop);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                impl->GetTypeString(), " does not support ", params->Signature());
      invoker->Run(arg.get(), StreamConfig{params->stream});
      return Status::OK();
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemm_op)));
  }
  return ret;
}

template <typename T, typename ALayout, typename BLayout>
auto GetCKStridedBatchedGemmTypeStringAndOps() {
  using CKDataType = typename CKDataTypeAdaptor<T>::type;
  using DeviceStridedBatchedGemm = ck::tensor_operation::device::DeviceBatchedGemm<
      ALayout, BLayout, Row,
      CKDataType, CKDataType, CKDataType,
      Nop, Nop, Nop>;
  using InstanceFactory =
      ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceStridedBatchedGemm>;

  std::vector<std::pair<std::string, Op<StridedBatchedGemmParams<T>>>> ret;
  for (auto&& impl : InstanceFactory::GetInstances()) {
    auto type_string = impl->GetTypeString();

    auto invoker = impl->MakeInvokerPointer();
    auto ck_gemm_op = [impl = std::move(impl), invoker = std::move(invoker)](const StridedBatchedGemmParams<T>* params) -> Status {
      auto one = ToHipType<T>::FromFloat(1.0f);
      auto zero = ToHipType<T>::FromFloat(0.0f);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          params->alpha != one || params->beta != zero,
          impl->GetTypeString(), " only supports alpha == 1 and beta == 0", params->Signature());

      auto nop = Nop{};
      auto arg = impl->MakeArgumentPointer(params->a, params->b, params->c,
                                           params->m, params->n, params->k,
                                           params->lda, params->ldb, params->ldc,
                                           params->stride_a, params->stride_b, params->stride_c,
                                           params->batch,
                                           nop, nop, nop);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                impl->GetTypeString(), " does not support ", params->Signature());
      invoker->Run(arg.get(), StreamConfig{params->stream});
      return Status::OK();
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemm_op)));
  }
  return ret;
}
#else
struct Row {};
struct Col {};
#endif  // USE_COMPOSABLE_KERNEL

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
