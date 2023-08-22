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
#include "core/providers/rocm/math/softmax_triton.cuh"

namespace onnxruntime {
namespace rocm {

template <typename InputT, typename OutputT, typename AccT, int VecSize>
Status SoftmaxBlockwiseOp(const SoftmaxParams<InputT, OutputT>* params) {
  dim3 grid(params->batch_count);
  dim3 block = SoftMax_getBlockSize(VecSize, params->softmax_elements);
  if (params->is_log_softmax) {
    softmax_block_forward<VecSize, InputT, AccT, OutputT, LogSoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(AccT), params->StreamHandle()>>>(
            params->output, const_cast<InputT*>(params->input),
            params->softmax_elements, params->input_stride,
            params->output_stride);
  } else {
    softmax_block_forward<VecSize, InputT, AccT, OutputT, SoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(AccT), params->StreamHandle()>>>(
            params->output, const_cast<InputT*>(params->input),
            params->softmax_elements, params->input_stride,
            params->output_stride);
  }
  return HIP_CALL(hipGetLastError());
}

template <typename InputT, typename OutputT, typename AccT>
Status SoftmaxWarpwiseStaticSelection(const SoftmaxParams<InputT, OutputT>* params) {
  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
      !(params->input_stride <= 1024 && params->input_stride * sizeof(InputT) <= 4096));
  if (params->softmax_elements == 0) {
    return Status::OK();
  } else {
    int log2_elements = log2_ceil(params->softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
    int warp_size = (next_power_of_two < GPU_WARP_SIZE_HOST) ? next_power_of_two : GPU_WARP_SIZE_HOST;

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
    int batches_per_warp = 1;
    // use 256 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 256;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (params->batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
#define LAUNCH_SOFTMAX_WARP_FORWARD(L2E)                              \
  case L2E:                                                           \
    softmax_warp_forward<InputT, OutputT, AccT, L2E>                  \
        <<<dim3(blocks), dim3(threads), 0, params->StreamHandle()>>>( \
            params->output, params->input, params->batch_count,       \
            params->input_stride, params->softmax_elements,           \
            params->is_log_softmax);                                  \
    break;
      LAUNCH_SOFTMAX_WARP_FORWARD(0);   // 1
      LAUNCH_SOFTMAX_WARP_FORWARD(1);   // 2
      LAUNCH_SOFTMAX_WARP_FORWARD(2);   // 4
      LAUNCH_SOFTMAX_WARP_FORWARD(3);   // 8
      LAUNCH_SOFTMAX_WARP_FORWARD(4);   // 16
      LAUNCH_SOFTMAX_WARP_FORWARD(5);   // 32
      LAUNCH_SOFTMAX_WARP_FORWARD(6);   // 64
      LAUNCH_SOFTMAX_WARP_FORWARD(7);   // 128
      LAUNCH_SOFTMAX_WARP_FORWARD(8);   // 256
      LAUNCH_SOFTMAX_WARP_FORWARD(9);   // 512
      LAUNCH_SOFTMAX_WARP_FORWARD(10);  // 1024
      default:
        break;
    }
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
        <<<grid, block, block.x * sizeof(AccT), params->StreamHandle()>>>(
            params->output, const_cast<InputT*>(params->input),
            params->softmax_elements, params->input_stride,
            params->output_stride);
  } else {
    softmax_block_forward<ILP, InputT, AccT, OutputT, SoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(AccT), params->StreamHandle()>>>(
            params->output, const_cast<InputT*>(params->input),
            params->softmax_elements, params->input_stride,
            params->output_stride);
  }
  return HIP_CALL(hipGetLastError());
}

template <typename InputT, typename OutputT, typename AccT>
Status SoftmaxStaticSelection(const SoftmaxParams<InputT, OutputT>* params) {
  auto status = SoftmaxWarpwiseStaticSelection<InputT, OutputT, AccT>(params);
  if (!status.IsOK()) {
    status = SoftmaxBlockwiseStaticSelection<InputT, OutputT, AccT>(params);
  }
  return status;
}

template <typename InputT, typename OutputT, typename AccT>
class SoftmaxTunableOp : public tunable::TunableOp<SoftmaxParams<InputT, OutputT>> {
 public:
  SoftmaxTunableOp() {
    this->RegisterOp(SoftmaxStaticSelection<InputT, OutputT, AccT>);
    this->RegisterOp(SoftmaxWarpwiseStaticSelection<InputT, OutputT, AccT>);
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

#ifdef USE_TRITON_KERNEL
    for (auto&& [_, op] : GetSoftmaxTritonOps<InputT, OutputT>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
    // this->RegisterOp(SoftmaxTritonOp<InputT, OutputT>);
#endif
  }
};

}  // namespace rocm
}  // namespace onnxruntime
