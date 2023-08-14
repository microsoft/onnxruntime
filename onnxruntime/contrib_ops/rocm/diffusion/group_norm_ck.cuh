// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#ifdef USE_COMPOSABLE_KERNEL
#include "core/providers/rocm/composable_kernel_common.h"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "contrib_ops/rocm/diffusion/group_norm_ck_impl/impl.cuh"
#endif  // USE_COMPOSABLE_KERNEL

#include "contrib_ops/rocm/diffusion/group_norm_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

#ifdef USE_COMPOSABLE_KERNEL

using onnxruntime::rocm::CKDataTypeAdaptor;

using Swish = ck::tensor_operation::element_wise::Swish;
using Pass = ck::tensor_operation::element_wise::PassThrough;

constexpr int Rank = 5;
constexpr int NumReduceDim = 3;

template <typename T, typename AccT, bool WithSwish>
auto GetCKGroupNormNHWCTypeStringAndOps() {
  using InDataType = typename CKDataTypeAdaptor<T>::type;
  using OutDataType = typename CKDataTypeAdaptor<T>::type;
  using AccDataType = typename CKDataTypeAdaptor<AccT>::type;
  using GammaDataType = float;
  using BetaDataType = float;

  using Activation = std::conditional_t<WithSwish, Swish, Pass>;

  std::vector<std::pair<std::string, onnxruntime::rocm::tunable::Op<GroupNormNHWCParams<T>>>> ret;
  for (auto&& impl : internal::GetDeviceGroupNormInstances<InDataType, GammaDataType, BetaDataType, AccDataType,
                                                           OutDataType, Activation, Rank, NumReduceDim>()) {
    std::string swish_suffix = WithSwish ? "_Swish" : "_Pass";
    auto type_string = onnxruntime::MakeString(impl->GetTypeString()) + swish_suffix;
    auto invoker = impl->MakeInvokerPointer();

    auto ck_group_norm_op = [impl = std::move(impl), invoker = std::move(invoker)](const GroupNormNHWCParams<T>* params) -> Status {
      if constexpr (WithSwish) {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
            !params->withSwish, "Swish version only support groupnorm with swish");
      } else {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
            params->withSwish, "Pass version only support groupnorm without swish");
      }
      std::vector<ck::index_t> in_lengths{params->n, params->h, params->w, params->groups, params->cPerGroup};
      std::vector<ck::index_t> in_out_strides{params->h * params->w * params->c, params->w * params->c, params->c, params->cPerGroup, 1};
      std::vector<ck::index_t> gamma_beta_strides{0, 0, 0, params->cPerGroup, 1};
      std::vector<ck::index_t> reduce_dims{1, 2, 4};

      auto activation = Activation{};

      auto arg = impl->MakeArgumentPointer(in_lengths,          // lengths
                                           in_out_strides,      // xStrides
                                           gamma_beta_strides,  // gammaStrides
                                           gamma_beta_strides,  // betaStrides
                                           in_out_strides,      // yStrides
                                           reduce_dims,         // reduceDims
                                           params->epsilon,
                                           params->src,
                                           params->gamma,
                                           params->beta,
                                           params->dst,
                                           nullptr,
                                           nullptr,
                                           activation);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                impl->GetTypeString(), " does not support ", params->Signature());
      invoker->Run(arg.get(), StreamConfig{params->StreamHandle()});
      return Status::OK();
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_group_norm_op)));
  }
  return ret;
}
#endif  // USE_COMPOSABLE_KERNEL

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
