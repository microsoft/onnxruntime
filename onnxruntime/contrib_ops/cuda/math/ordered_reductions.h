// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

class OrderedRowReduce final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit OrderedRowReduce(const OpKernelInfo& info);
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OrderedRowReduce);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int width_;
  bool normalize_;
};

class OrderedMatrixReduce128 final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit OrderedMatrixReduce128(const OpKernelInfo& info) : CudaKernel(info) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OrderedMatrixReduce128);
  Status ComputeInternal(OpKernelContext* context) const override;
};

Status LaunchOrderedRowReduce(cudaStream_t stream, const float* input, float* output,
                              size_t rows, int width, bool normalize);
Status LaunchOrderedMatrixReduce128(cudaStream_t stream, const float* input, float* output, size_t matrices);

}  // namespace onnxruntime::contrib::cuda
