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

// The SiLU function is a special case of Swish function,
// The Swish function is parametrized by b, which is set to 1.0 for SiLU. They are defined as:
// SiLU(x) = x * sigmoid(x)
// Swish(x) = x * sigmoid(bx)
// The default value of b is 1.0 in ck::tensor_operation::element_wise::Swish function. We treat them as the same function here.
using Silu = ck::tensor_operation::element_wise::Swish;
using Pass = ck::tensor_operation::element_wise::PassThrough;

constexpr int Rank = 5;
constexpr int NumReduceDim = 3;

template <typename T, typename AccT, bool WithSilu>
auto GetCKGroupNormNHWCTypeStringAndOps() {
  using XDataType = typename CKDataTypeAdaptor<T>::type;
  using YDataType = typename CKDataTypeAdaptor<T>::type;
  using SaveMeanInvStdDataType = typename CKDataTypeAdaptor<AccT>::type;
  using GammaDataType = float;
  using BetaDataType = float;

  using Activation = std::conditional_t<WithSilu, Silu, Pass>;

  std::vector<std::pair<std::string, onnxruntime::rocm::tunable::Op<GroupNormNHWCTunableParams<T>>>> ret;
  for (auto&& impl : internal::GetDeviceGroupNormInstances<XDataType, GammaDataType, BetaDataType, YDataType,
                                                           SaveMeanInvStdDataType, Activation, Rank, NumReduceDim>()) {
    std::string silu_suffix = WithSilu ? "_Silu" : "_Pass";
    auto type_string = onnxruntime::MakeString(impl->GetTypeString()) + silu_suffix;
    auto invoker = impl->MakeInvokerPointer();

    auto ck_group_norm_op = [impl = std::move(impl), invoker = std::move(invoker)](
                                const GroupNormNHWCTunableParams<T>* params) -> Status {
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF((params->skip != nullptr || params->bias != nullptr),
                                                "Input skip or bias is not supported by composable kernel.");
      if constexpr (WithSilu) {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
            !params->use_silu, "Silu version only support groupnorm with silu");
      } else {
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
            params->use_silu, "Pass version only support groupnorm without silu");
      }
      std::vector<ck::index_t> in_lengths{params->n, params->h, params->w, params->groups, params->channels_per_group};
      std::vector<ck::index_t> in_out_strides{params->h * params->w * params->c, params->w * params->c,
                                              params->c, params->channels_per_group, 1};
      std::vector<ck::index_t> gamma_beta_strides{0, 0, 0, params->channels_per_group, 1};
      std::vector<ck::index_t> reduce_dims{1, 2, 4};

      auto activation = Activation{};

      auto arg = impl->MakeArgumentPointer(in_lengths,          // lengths
                                           in_out_strides,      // xStrides
                                           gamma_beta_strides,  // gammaStrides
                                           gamma_beta_strides,  // betaStrides
                                           in_out_strides,      // yStrides
                                           {0, 0},              // saveMeanStrides
                                           {0, 0},              // saveInvStdStrides
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
                                                impl->GetTypeString(), " does not support the params");
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
