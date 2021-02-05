// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/generator/constant_of_shape_base.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

class ConstantOfShape final : public ConstantOfShapeBase<>, public CudaKernel {
 public:
  explicit ConstantOfShape(const OpKernelInfo& info) : ConstantOfShapeBase(info), CudaKernel(info) {}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConstantOfShape);

  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
