// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/math/gemm.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

class DummyGemm final : public RocmKernel {
 public:
  DummyGemm(const OpKernelInfo& op_kernel_info) : RocmKernel(op_kernel_info), gemm_float_(op_kernel_info), gemm_half_(op_kernel_info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
 private:
  Gemm<float> gemm_float_;
  Gemm<MLFloat16> gemm_half_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
