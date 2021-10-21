// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "gsl/gsl"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/transpose.h"

namespace onnxruntime {
namespace cuda {

class Transpose final : public CudaKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : CudaKernel(info), TransposeBase(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

  static Status DoTranspose(const Transpose& transpose_kernel,
                            const std::vector<size_t>& permutations, const Tensor& input, Tensor& output);

  //  `input_shape_override` (if provided) overrides the shape of `input` for compute purposes
  static Status DoTranspose(const cudaDeviceProp& prop,
                            cudaStream_t stream,
                            const cublasHandle_t cublas_handle,
                            const std::vector<size_t>& permutations,
                            const Tensor& input, Tensor& output, const TensorShape* input_shape_override = nullptr);
};

}  // namespace cuda
}  // namespace onnxruntime
