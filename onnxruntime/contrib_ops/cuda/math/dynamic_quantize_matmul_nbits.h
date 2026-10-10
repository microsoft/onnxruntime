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
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DynamicQuantizeMatMulNBits);

  int64_t bits_;
  int64_t block_size_;
  int64_t activation_block_size_;
  DynamicQuantizationMode mode_;
  bool constant_scales_validated_{false};
};

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
