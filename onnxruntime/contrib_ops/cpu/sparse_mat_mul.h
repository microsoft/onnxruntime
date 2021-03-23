// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// This class implements Sparse Matrix Multiplication MatMul
// Current support is SmMM Sparse To Dense Matrix with Dense Output
// 2-Dim, not batching support, no Fused optimization so no integrated
// transpose
class SparseMatMul final : public OpKernel {
 public:
  SparseMatMul(const OpKernelInfo& info) : OpKernel(info) {
  }
  common::Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
