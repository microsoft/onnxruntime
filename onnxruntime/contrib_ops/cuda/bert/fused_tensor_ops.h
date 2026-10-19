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
Status LaunchEmbeddingRMSNorm(cudaStream_t stream, const int64_t* ids, const uint16_t* table,
                              const float* weight, float epsilon, uint16_t* output,
                              int64_t rows, int64_t vocab_size, int64_t hidden_size);
Status LaunchRowSelect(cudaStream_t stream, const int64_t* selection,
                       const uint16_t* const* inputs, uint16_t* const* outputs,
                       const int64_t* row_sizes);
Status LaunchSigmoidGateMul(cudaStream_t stream, const uint16_t* attention,
                            const uint16_t* projection, uint16_t* output, int64_t count,
                            int head_dim, int heads, int q_stride,
                            int64_t token_stride, int64_t q_offset,
                            int64_t gate_row_width, const uint16_t* residual, bool sigmoid_in_fp32,
                            bool silu_gate);
Status LaunchDequantizeLinearFp8Bf16(cudaStream_t stream, const uint8_t* input,
                                     const uint8_t* zero_point, const float* scales,
                                     uint16_t* output, int64_t count,
                                     int64_t axis_size, int64_t inner_size,
                                     bool scalar_zero_point);
Status LaunchUntileHyperConnectionWeights(cudaStream_t stream, const uint16_t* input,
                                          uint16_t* output, int kind);

template <typename T>
class ArgMaxLastAxis final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit ArgMaxLastAxis(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool keepdims_;
};

class EmbeddingRMSNorm final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit EmbeddingRMSNorm(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
};

class RowSelect final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit RowSelect(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
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

class DequantizeLinearFp8Bf16 final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit DequantizeLinearFp8Bf16(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

class UntileHyperConnectionWeights final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit UntileHyperConnectionWeights(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int kind_;
};

}  // namespace onnxruntime::contrib::cuda
