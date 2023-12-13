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
#include "ck/library/tensor_operation_instance/gpu/gemm_splitk.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_streamk.hpp"
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

template <typename T, BlasOp OpA, BlasOp OpB>
auto GetCKGemmTypeStringAndOps() {
  using CKDataType = typename CKDataTypeAdaptor<T>::type;
  using ALayout = typename CKBlasOpAdaptor<OpA>::type;
  using BLayout = typename CKBlasOpAdaptor<OpB>::type;
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
      auto one = ToHipType<T>::FromFloat(1.0f);
      auto zero = ToHipType<T>::FromFloat(0.0f);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(params->alpha != one || params->beta != zero,
                                                impl->GetTypeString(), " only supports alpha == 1 and beta == 0");

      auto nop = Nop{};
      auto arg = impl->MakeArgumentPointer(params->a, params->b, params->c,
                                           params->m, params->n, params->k,
                                           params->lda, params->ldb, params->ldc,
                                           nop, nop, nop);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                impl->GetTypeString(), " does not support the params");
      invoker->Run(arg.get(), StreamConfig{params->StreamHandle()});
      return Status::OK();
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemm_op)));
  }
  return ret;
}

template <typename T, BlasOp OpA, BlasOp OpB>
auto GetCKStreamKGemmTypeStringAndOps() {
  using CKDataType = typename CKDataTypeAdaptor<T>::type;
  using ALayout = typename CKBlasOpAdaptor<OpA>::type;
  using BLayout = typename CKBlasOpAdaptor<OpB>::type;
  using DeviceGemm = ck::tensor_operation::device::DeviceGemmStreamK<
      ALayout, BLayout, Row,
      CKDataType, CKDataType, CKDataType,
      Nop, Nop, Nop>;
  using InstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemm>;

  std::vector<std::pair<std::string, Op<GemmParams<T>>>> ret;
  for (auto&& impl : InstanceFactory::GetInstances()) {
    auto type_string = impl->GetTypeString();
    auto invoker = impl->MakeInvokerPointer();
    auto ck_gemm_op = [impl = std::move(impl), invoker = std::move(invoker)](const GemmParams<T>* params) -> Status {
      auto one = ToHipType<T>::FromFloat(1.0f);
      auto zero = ToHipType<T>::FromFloat(0.0f);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(params->alpha != one || params->beta != zero,
                                                impl->GetTypeString(), " only supports alpha == 1 and beta == 0");

      auto nop = Nop{};
      auto arg = impl->MakeArgumentPointer(params->a, params->b, params->c,
                                           params->m, params->n, params->k,
                                           params->lda, params->ldb, params->ldc,
                                           nop, nop, nop);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                impl->GetTypeString(), " does not support ", params->Signature());
      invoker->Run(arg.get(), StreamConfig{params->StreamHandle()});
      return Status::OK();
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemm_op)));
  }
  return ret;
}

template <typename T, BlasOp OpA, BlasOp OpB>
auto GetCKSplitKGemmTypeStringAndOps() {
  using CKDataType = typename CKDataTypeAdaptor<T>::type;
  using ALayout = typename CKBlasOpAdaptor<OpA>::type;
  using BLayout = typename CKBlasOpAdaptor<OpB>::type;
  using DeviceGemm = ck::tensor_operation::device::DeviceGemmSplitK<
      ALayout, BLayout, Row,
      CKDataType, CKDataType, CKDataType,
      Nop, Nop, Nop>;
  using InstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemm>;

  std::vector<std::pair<std::string, Op<GemmParams<T>>>> ret;
  for (auto num_split : {4, 16, 64}) {
    auto instances = InstanceFactory::GetInstances();
    for (auto&& impl : instances) {
      auto type_string = impl->GetTypeString() + "_SplitK" + std::to_string(num_split);
      auto invoker = impl->MakeInvokerPointer();
      auto ck_gemm_op = [num_split, impl = std::move(impl), invoker = std::move(invoker)](const GemmParams<T>* params) -> Status {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
            params->k < 128 * num_split, "k=", params->k, " is too small, it makes no sense to use this split-k gemm.");

        auto one = ToHipType<T>::FromFloat(1.0f);
        auto zero = ToHipType<T>::FromFloat(0.0f);
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(params->alpha != one || params->beta != zero,
                                                  impl->GetTypeString(), " only supports alpha == 1 and beta == 0");

        auto nop = Nop{};
        auto arg = impl->MakeArgumentPointer(params->a, params->b, params->c,
                                             params->m, params->n, params->k,
                                             params->lda, params->ldb, params->ldc,
                                             nop, nop, nop, num_split);
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                  impl->GetTypeString(), " does not support ", params->Signature());
        invoker->Run(arg.get(), StreamConfig{params->StreamHandle()});
        return Status::OK();
      };
      ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemm_op)));
    }
  }
  return ret;
}

template <typename T, BlasOp OpA, BlasOp OpB>
auto GetCKStridedBatchedGemmTypeStringAndOps() {
  using CKDataType = typename CKDataTypeAdaptor<T>::type;
  using ALayout = typename CKBlasOpAdaptor<OpA>::type;
  using BLayout = typename CKBlasOpAdaptor<OpB>::type;
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
          impl->GetTypeString(), " only supports alpha == 1 and beta == 0");

      auto nop = Nop{};
      auto arg = impl->MakeArgumentPointer(params->a, params->b, params->c,
                                           params->m, params->n, params->k,
                                           params->lda, params->ldb, params->ldc,
                                           params->stride_a, params->stride_b, params->stride_c,
                                           params->batch,
                                           nop, nop, nop);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                impl->GetTypeString(), " does not support the params");
      invoker->Run(arg.get(), StreamConfig{params->StreamHandle()});
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
