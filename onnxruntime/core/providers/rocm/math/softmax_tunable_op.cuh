// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/math/softmax_common.h"
#include "core/providers/rocm/math/softmax_ck.cuh"
#include "core/providers/rocm/math/softmax_warpwise_impl.cuh"
#include "core/providers/rocm/math/softmax_blockwise_impl.cuh"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace rocm {

template <typename input_t, typename output_t, typename acc_t, int VecSize>
Status SoftmaxBlockwiseOp(const SoftmaxParams<input_t, output_t>* params) {
  dim3 grid(params->batch_count);
  dim3 block = SoftMax_getBlockSize(VecSize, params->softmax_elements);
  if (params->is_log_softmax) {
    softmax_block_forward<VecSize, input_t, acc_t, output_t, LogSoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(acc_t), params->stream>>>(params->output, const_cast<input_t*>(params->input),
                                                                   params->softmax_elements, params->input_stride,
                                                                   params->output_stride);
  } else {
    softmax_block_forward<VecSize, input_t, acc_t, output_t, SoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(acc_t), params->stream>>>(params->output, const_cast<input_t*>(params->input),
                                                                   params->softmax_elements, params->input_stride,
                                                                   params->output_stride);
  }
  return HIP_CALL(hipGetLastError());
}

template <typename input_t, typename output_t, typename acc_t>
Status SoftmaxBlockwiseStaticSelection(const SoftmaxParams<input_t, output_t>* params) {
  dim3 grid(params->batch_count);
  constexpr int ILP = sizeof(float4) / sizeof(input_t);
  dim3 block = SoftMax_getBlockSize(ILP, params->softmax_elements);
  if (params->is_log_softmax) {
    softmax_block_forward<ILP, input_t, acc_t, output_t, LogSoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(acc_t), params->stream>>>(params->output, const_cast<input_t*>(params->input),
                                                                   params->softmax_elements, params->input_stride,
                                                                   params->output_stride);
  } else {
    softmax_block_forward<ILP, input_t, acc_t, output_t, SoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(acc_t), params->stream>>>(params->output, const_cast<input_t*>(params->input),
                                                                   params->softmax_elements, params->input_stride,
                                                                   params->output_stride);
  }
  return HIP_CALL(hipGetLastError());
}

template <typename input_t, typename output_t, typename acc_t>
class SoftmaxTunableOp : public onnxruntime::rocm::tunable::TunableOp<SoftmaxParams<input_t, output_t>> {
 public:
  SoftmaxTunableOp() {
    this->RegisterOp(SoftmaxBlockwiseStaticSelection<input_t, output_t, acc_t>);
    this->RegisterOp(SoftmaxBlockwiseOp<input_t, output_t, acc_t, 1>);
    this->RegisterOp(SoftmaxBlockwiseOp<input_t, output_t, acc_t, 2>);
    this->RegisterOp(SoftmaxBlockwiseOp<input_t, output_t, acc_t, 4>);
    this->RegisterOp(SoftmaxBlockwiseOp<input_t, output_t, acc_t, 8>);
    this->RegisterOp(SoftmaxBlockwiseOp<input_t, output_t, acc_t, 16>);

#ifdef USE_COMPOSABLE_KERNEL
    for (auto&& [_, op] : GetCKSoftmaxTypeStringAndOps<input_t, output_t, acc_t>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif  // USE_COMPOSABLE_KERNEL

    // NOTE: the 1st kernel is SoftmaxBlockwise Original implementation.
    this->SetDefaultId(0);
  }
};

}  // namespace rocm
}  // namespace onnxruntime
