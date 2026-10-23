// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

Status LaunchArgMaxLastAxis(cudaStream_t stream, const void* input, int input_type,
                            int64_t rows, int64_t width, int64_t* output,
                            float* partial_values, int64_t* partial_indices);
inline int64_t GetArgMaxLastAxisPartitions(int64_t width) {
  return 1 + (width - 1) / 4096;
}
Status LaunchSigmoidGateMul(cudaStream_t stream, const uint16_t* attention,
                            const uint16_t* projection, uint16_t* output, int64_t count,
                            int head_dim, int heads, int q_stride,
                            int64_t token_stride, int64_t q_offset,
                            int64_t gate_row_width, const uint16_t* residual, bool sigmoid_in_fp32,
                            bool silu_gate);
Status LaunchProjectedSigmoidGateMul(cudaStream_t stream, const uint16_t* attention,
                                     const uint16_t* activation, const uint16_t* weight,
                                     const uint16_t* residual, uint16_t* output,
                                     int64_t rows, int64_t input_width, int64_t output_width,
                                     bool sigmoid_in_fp32);
Status LaunchPrepareSharedExpert(cudaStream_t stream, const uint16_t* up, const uint16_t* silu_gate,
                                 const uint16_t* activation, const uint16_t* weight,
                                 uint16_t* prepared, float* row_gates,
                                 int rows, int width, int input_width);

template <typename T>
class ArgMaxLastAxis final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit ArgMaxLastAxis(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool keepdims_;
};

class SigmoidGateMul final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit SigmoidGateMul(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int head_dim_;
  int heads_;
  int q_stride_;
  int64_t tok_stride_;
  int64_t q_offset_;
  bool broadcast_gate_;
  bool sigmoid_in_fp32_;
  bool silu_gate_;
};

}  // namespace onnxruntime::contrib::cuda
