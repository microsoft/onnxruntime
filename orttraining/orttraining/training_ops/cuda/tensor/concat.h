// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/concat.h"

namespace onnxruntime {
namespace cuda {

class ConcatTraining final : public CudaKernel, public ConcatBase {
 public:
  ConcatTraining(const OpKernelInfo& info) : CudaKernel(info), ConcatBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
