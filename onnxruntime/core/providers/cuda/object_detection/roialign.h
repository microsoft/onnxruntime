// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/object_detection/roialign.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
struct RoiAlign final : CudaKernel, RoiAlignBase {
  RoiAlign(const OpKernelInfo& info) : CudaKernel(info), RoiAlignBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RoiAlign);
};
}  // namespace cuda
}  // namespace onnxruntime
