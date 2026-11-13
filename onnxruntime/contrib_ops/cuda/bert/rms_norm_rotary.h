// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"
namespace onnxruntime::contrib::cuda {
Status LaunchRMSNormRotary(cudaStream_t stream, const uint16_t* input, const float* scale,
                           const int64_t* positions, const void* cosine, const void* sine,
                           void* output, float* normalized, int groups, int heads, int sequence,
                           int input_stride, int width, int rotary_dim, int cache_rows, bool scalar_position,
                           float epsilon, bool dense_prefill, bool fp32_rotary,
                           bool fp32_cos = false, bool output_fp16 = false, bool input_fp16 = false);

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
  bool output_fp16_;
};

}  // namespace onnxruntime::contrib::cuda
