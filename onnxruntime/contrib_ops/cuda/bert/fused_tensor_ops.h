// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

Status LaunchRMSNormRotary(cudaStream_t stream, const uint16_t* input, const float* scale,
                          const int64_t* positions, const uint16_t* cosine, const uint16_t* sine,
                          uint16_t* output, float* normalized, int groups, int heads, int sequence,
                          int input_stride, int rotary_dim, int cache_rows, bool scalar_position,
                          float epsilon, bool dense_prefill, bool fp32_rotary);

class RMSNormRotary final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit RMSNormRotary(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
  int64_t input_start_;
  int64_t input_width_;
  int64_t rotary_dim_;
  bool fp32_rotary_;
  bool flatten_batch_;
};

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
                       const int64_t* row_sizes, int64_t rows);
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
Status LaunchDotProductGate(cudaStream_t stream, const uint16_t* key, const uint16_t* query,
                            const uint16_t* value, uint16_t* output, int64_t groups,
                            int64_t width, int64_t value_repeat, float divisor, float epsilon);
Status LaunchDequantizeLinearFp8Bf16(cudaStream_t stream, const uint8_t* input,
                                     const uint8_t* zero_point, const float* scales,
                                     uint16_t* output, int64_t count,
                                     int64_t axis_size, int64_t inner_size,
                                     bool scalar_zero_point);
Status LaunchUntileHyperConnectionWeights(cudaStream_t stream, const uint16_t* input,
                                          uint16_t* output, int kind);
Status LaunchMtpProjectionAdd(cudaStream_t stream, const uint16_t* hidden, const uint16_t* embedding,
                              uint16_t* output, int64_t count, int64_t width, int64_t branches);
Status LaunchPackedMtpMoe(cudaStream_t stream, const uint16_t* input, const uint16_t* router,
                          const uint8_t* gate_up, const uint8_t* down, uint16_t* output,
                          float* scratch, int rows, float gate_up_scale, float down_scale);

class PackedMtpMoe final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit PackedMtpMoe(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;
 private:
  float gate_up_scale_;
  float down_scale_;
};

class MtpInputProjection final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MtpInputProjection(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

class HyperConnectionMixer final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit HyperConnectionMixer(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;
 private:
  int64_t branches_;
  float alpha_;
  float reduction_scale_;
};

class MtpAttentionProjection final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MtpAttentionProjection(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;
 private:
  int64_t heads_;
  int64_t rotary_dim_;
  float epsilon_;
};

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

class DotProductGate final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit DotProductGate(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t width_;
  float divisor_;
  float epsilon_;
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
