// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/math/softmax_ck.cuh"
#include "core/providers/rocm/math/softmax_common.h"
#include "core/providers/rocm/math/softmax_warpwise_impl.cuh"
#include "core/providers/rocm/math/softmax_blockwise_impl.cuh"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace rocm {

template <typename InputT, typename OutputT, typename AccT, int VecSize>
Status SoftmaxBlockwiseOp(const SoftmaxParams<InputT, OutputT>* params) {
  dim3 grid(params->batch_count);
  dim3 block = SoftMax_getBlockSize(VecSize, params->softmax_elements);
  if (params->is_log_softmax) {
    softmax_block_forward<VecSize, InputT, AccT, OutputT, LogSoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(AccT), params->stream>>>(params->output, const_cast<InputT*>(params->input),
                                                                  params->softmax_elements, params->input_stride,
                                                                  params->output_stride);
  } else {
    softmax_block_forward<VecSize, InputT, AccT, OutputT, SoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(AccT), params->stream>>>(params->output, const_cast<InputT*>(params->input),
                                                                  params->softmax_elements, params->input_stride,
                                                                  params->output_stride);
  }
  return HIP_CALL(hipGetLastError());
}

template <typename InputT, typename OutputT, typename AccT>
Status SoftmaxBlockwiseStaticSelection(const SoftmaxParams<InputT, OutputT>* params) {
  dim3 grid(params->batch_count);
  constexpr int ILP = sizeof(float4) / sizeof(InputT);
  dim3 block = SoftMax_getBlockSize(ILP, params->softmax_elements);
  if (params->is_log_softmax) {
    softmax_block_forward<ILP, InputT, AccT, OutputT, LogSoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(AccT), params->stream>>>(params->output, const_cast<InputT*>(params->input),
                                                                  params->softmax_elements, params->input_stride,
                                                                  params->output_stride);
  } else {
    softmax_block_forward<ILP, InputT, AccT, OutputT, SoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(AccT), params->stream>>>(params->output, const_cast<InputT*>(params->input),
                                                                  params->softmax_elements, params->input_stride,
                                                                  params->output_stride);
  }
  return HIP_CALL(hipGetLastError());
}

template <typename InputT, typename OutputT, typename AccT>
class SoftmaxTunableOp : public onnxruntime::rocm::tunable::TunableOp<SoftmaxParams<InputT, OutputT>> {
 public:
  SoftmaxTunableOp() {
    this->RegisterOp(SoftmaxBlockwiseStaticSelection<InputT, OutputT, AccT>);
    this->RegisterOp(SoftmaxBlockwiseOp<InputT, OutputT, AccT, 1>);
    this->RegisterOp(SoftmaxBlockwiseOp<InputT, OutputT, AccT, 2>);
    this->RegisterOp(SoftmaxBlockwiseOp<InputT, OutputT, AccT, 4>);
    this->RegisterOp(SoftmaxBlockwiseOp<InputT, OutputT, AccT, 8>);
    this->RegisterOp(SoftmaxBlockwiseOp<InputT, OutputT, AccT, 16>);

#ifdef USE_COMPOSABLE_KERNEL
    for (auto&& [_, op] : GetCKSoftmaxTypeStringAndOps<InputT, OutputT, AccT>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif  // USE_COMPOSABLE_KERNEL
  }
};

}  // namespace rocm
}  // namespace onnxruntime
