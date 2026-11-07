// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

class UnitLowerTriangularSolve final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit UnitLowerTriangularSolve(const OpKernelInfo& info);
  ~UnitLowerTriangularSolve() override;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(UnitLowerTriangularSolve);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace onnxruntime::contrib::cuda
