// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

enum class DynamicQuantizationMode { PerTensor,
                                     PerToken,
                                     PerBlock };
enum class DynamicWeightScaleType { Float,
                                    Half,
                                    BFloat16 };

class DynamicQuantizeMatMulNBits final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit DynamicQuantizeMatMulNBits(const OpKernelInfo& info);
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DynamicQuantizeMatMulNBits);

  int64_t bits_;
  int64_t block_size_;
  int64_t activation_block_size_;
  DynamicQuantizationMode mode_;
  bool constant_scales_validated_{false};
  bool enable_sm90_{true};
  IAllocatorUniquePtr<int8_t> signed_weights_;
  int64_t signed_n_{0};
  int64_t signed_stride_{0};
  bool signed_grouped_{false};
  int sm90_device_id_{0};
  int sm90_sm_count_{0};
};

Status DynamicQuantizedSm90Available(bool& available);
Status LaunchDynamicExpandedWeights(const uint8_t* weights, int8_t* expanded, int bits, int64_t count,
                                    cudaStream_t stream);
Status LaunchDynamicSignedWeights(const uint8_t* weights, int8_t* signed_weights, int64_t count,
                                  cudaStream_t stream);
Status LaunchDynamicQuantizedSm90(const int8_t* a, const float* a_scales, const int8_t* b,
                                  const void* b_scales, DynamicWeightScaleType scale_type,
                                  const void* bias, void* y, bool bf16, bool per_tensor, int rows, int n, int k,
                                  int64_t b_stride, int device_id, int sm_count, cudaStream_t stream);
Status LaunchDynamicQuantizedSm90Grouped(const int8_t* a, const float* a_scales, const int8_t* b,
                                         const void* b_scales, DynamicWeightScaleType scale_type,
                                         const void* bias, void* y, bool bf16, bool per_tensor, int rows, int n,
                                         int k, int sm_count, cudaStream_t stream);
Status LaunchValidateDynamicWeightScales(const void* scales, DynamicWeightScaleType type,
                                         int64_t count, int* invalid, cudaStream_t stream);
// Number of float partial maxima needed by the bounded tensor-wide reduction.
size_t DynamicTensorScaleScratchSize(int64_t count);
Status LaunchDynamicTensorScale(const void* a, bool bf16, int64_t count,
                                float* scale, float* partials, cudaStream_t stream);
Status LaunchDynamicQuantizeRows(const void* a, bool bf16, int64_t rows, int64_t k,
                                 int64_t activation_block_size, const float* tensor_scale,
                                 int8_t* quantized, float* scales, cudaStream_t stream);
Status LaunchDynamicQuantizedMatMul(const int8_t* a, const float* a_scales,
                                    const uint8_t* b, const void* b_scales, DynamicWeightScaleType scale_type,
                                    const void* bias, void* y, bool bf16, bool per_tensor,
                                    int64_t rows, int64_t n, int64_t k,
                                    int bits, int64_t block_size, int64_t activation_block_size,
                                    int compute_capability, cudaStream_t stream);
Status LaunchDynamicQuantizedEmpty(void* y, const void* bias, bool bf16,
                                   int64_t count, int64_t n, cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda
