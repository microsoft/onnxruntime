// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/math/softmax_warpwise_impl.cuh"
#include "core/providers/rocm/math/softmax_blockwise_impl.cuh"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace rocm {

template <typename input_t, typename output_t>
struct SoftmaxParams : onnxruntime::rocm::tunable::OpParams {
  SoftmaxParams(hipStream_t stream, output_t* output, const input_t* input, int softmax_elements,
                int input_stride, int output_stride, int batch_count, bool is_log_softmax)
      : OpParams(stream), output(output), input(input), softmax_elements(softmax_elements), input_stride(input_stride),
        output_stride(output_stride), batch_count(batch_count), is_log_softmax(is_log_softmax) {}

  std::string Signature() const override {
    std::string sig = std::to_string(batch_count) + "_" + std::to_string(softmax_elements);
    return sig;
  }

  output_t* output;
  const input_t* input;
  int softmax_elements;
  int input_stride;
  int output_stride;
  int batch_count;
  bool is_log_softmax;
};

template <typename input_t, typename output_t, typename acc_t, int VecSize>
Status SoftmaxBlockwiseOp(SoftmaxParams<input_t, output_t>* params) {
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

}  // namespace rocm
}  // namespace onnxruntime
