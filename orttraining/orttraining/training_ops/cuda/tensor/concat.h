// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/concatbase.h"

namespace onnxruntime {
namespace cuda {

class ConcatTraining final : public CudaKernel, public ConcatBase {
 public:
  ConcatTraining(const OpKernelInfo& info) : CudaKernel(info), ConcatBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
