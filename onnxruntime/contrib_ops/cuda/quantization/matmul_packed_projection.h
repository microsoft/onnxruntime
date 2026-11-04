// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/quantization/matmul_packed_projection.cuh"

namespace onnxruntime::contrib::cuda {

template <PackedProjectionFormat Format>
class MatMulPackedProjection final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulPackedProjection(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulPackedProjection);
};

using MatMulPackedNVFP4 = MatMulPackedProjection<PackedProjectionFormat::NVFP4>;
using MatMulPackedQ8 = MatMulPackedProjection<PackedProjectionFormat::Q8>;
using MatMulPackedIQ2XXS = MatMulPackedProjection<PackedProjectionFormat::IQ2XXS>;

}  // namespace onnxruntime::contrib::cuda
