// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#ifdef USE_COMPOSABLE_KERNEL
#include "core/providers/rocm/composable_kernel_common.h"

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax.hpp"
#include "ck/tensor_operation/gpu/device/device_softmax.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#endif  // USE_COMPOSABLE_KERNEL

#include "core/providers/rocm/math/softmax_common.h"

namespace onnxruntime {
namespace rocm {

#ifdef USE_COMPOSABLE_KERNEL

using Nop = ck::tensor_operation::element_wise::PassThrough;
constexpr int Rank = 4;
constexpr int NumReduceDim = 1;

template <typename InputT, typename OutputT, typename AccT>
auto GetCKSoftmaxTypeStringAndOps() {
  using InDataType = typename CKDataTypeAdaptor<InputT>::type;
  using OutDataType = typename CKDataTypeAdaptor<OutputT>::type;
  using AccDataType = typename CKDataTypeAdaptor<AccT>::type;
  using DeviceSoftmax = ck::tensor_operation::device::
      DeviceSoftmax<InDataType, AccDataType, OutDataType, Nop, Nop, Rank>;
  using InstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceSoftmax>;

  std::vector<std::pair<std::string, tunable::Op<SoftmaxParams<InputT, OutputT>>>> ret;
  for (auto&& impl : InstanceFactory::GetInstances()) {
    auto type_string = onnxruntime::MakeString(impl->GetTypeString());
    auto invoker = impl->MakeInvokerPointer();

    auto ck_softmax_op = [impl = std::move(impl), invoker = std::move(invoker)](const SoftmaxParams<InputT, OutputT>* params) -> Status {
      double alpha{1.0f};
      double beta{0.0f};

      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          params->is_log_softmax,
          impl->GetTypeString(), " does not support log softmax");
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          impl->GetRank() != Rank || impl->GetNumReduceDim() != NumReduceDim,
          impl->GetTypeString(), " does not support current Rank or NumReduceDim ", params->Signature());

      std::vector<ck::index_t> in_lengths{1, 1, params->batch_count, params->softmax_elements};
      std::vector<ck::index_t> in_strides{params->batch_count * params->input_stride, params->batch_count * params->input_stride, params->input_stride, 1};
      std::vector<ck::index_t> reduce_dims{3};

      auto nop = Nop{};
      auto arg = impl->MakeArgumentPointer(in_lengths, in_strides, reduce_dims, alpha, beta,
                                           params->input, params->output, nop, nop);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                impl->GetTypeString(), " does not support ", params->Signature());
      invoker->Run(arg.get(), StreamConfig{params->stream});
      return Status::OK();
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_softmax_op)));
  }
  return ret;
}
#endif  // USE_COMPOSABLE_KERNEL

}  // namespace rocm
}  // namespace onnxruntime
