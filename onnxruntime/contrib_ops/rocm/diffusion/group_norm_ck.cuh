// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdio.h>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_COMPOSABLE_KERNEL
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

template <typename T>
struct DataTypeAdaptor {
  using type = T;
};

template <>
struct DataTypeAdaptor<half> {
  using type = ck::half_t;
};

template <>
struct DataTypeAdaptor<BFloat16> {
  using type = ck::bhalf16_t;
};

using Swish = ck::tensor_operation::element_wise::Swish;
constexpr int Rank = 5;
constexpr int NumReduceDim = 3;

template <typename T, typename AccT>
auto GetCKGroupNormNHWCTypeStringAndOps() {
  using InDataType = typename DataTypeAdaptor<T>::type;
  using OutDataType = typename DataTypeAdaptor<T>::type;
  using AccDataType = typename DataTypeAdaptor<AccT>::type;
  using GammaDataType = float;
  using BetaDataType = float;

  std::vector<std::pair<std::string, onnxruntime::rocm::tunable::Op<GroupNormNHWCParams<T>>>> ret;
  for (auto&& impl : internal::GetDeviceGroupNormInstances<InDataType, GammaDataType, BetaDataType, AccDataType,
                                                           OutDataType, Swish, Rank, NumReduceDim>()) {
    auto type_string = onnxruntime::MakeString(impl->GetTypeString());
    auto invoker = impl->MakeInvokerPointer();

    auto ck_group_norm_op = [impl = std::move(impl), invoker = std::move(invoker)](const GroupNormNHWCParams<T>* params) -> Status {
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          !params->withSwish,
          impl->GetTypeString(), " only supports group norm with swish");

      std::vector<ck::index_t> in_lengths{params->n, params->h, params->w, params->groups, params->cPerGroup};
      std::vector<ck::index_t> in_out_strides{params->h * params->w * params->c, params->w * params->c, params->c, params->cPerGroup, 1};
      std::vector<ck::index_t> gamma_beta_strides{0, 0, 0, params->cPerGroup, 1};
      std::vector<ck::index_t> reduce_dims{1, 2, 4};

      auto swish = Swish{};
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
                                           swish);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                impl->GetTypeString(), " does not support ", params->Signature());
      invoker->Run(arg.get(), StreamConfig{params->stream});
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
