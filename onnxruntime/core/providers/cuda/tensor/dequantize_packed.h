// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/quantization/dequantize_packed.h"

namespace onnxruntime {
namespace cuda {

class DequantizePacked final : public CudaKernel {
 public:
  explicit DequantizePacked(const OpKernelInfo& info) : CudaKernel(info), attributes_(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  dequantize_packed::Attributes attributes_;
};

Status LaunchDequantizePacked(cudaStream_t stream, const uint8_t* input, void* output,
                              int format, int output_dtype, int64_t elements);

}  // namespace cuda
}  // namespace onnxruntime
