// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <memory>

struct MLAS_GEMM_PARAMETERS;

namespace onnxruntime {

template<typename T>
class PackForGemm final : public OpKernel {
 public:
  explicit PackForGemm(const OpKernelInfo& info);

  ~PackForGemm();

  Status Compute(OpKernelContext* context) const override;
 private:
  std::unique_ptr<MLAS_GEMM_PARAMETERS> gemm_params_;
};

template<typename T>
class MatMulPrepacked final : public OpKernel {
 public:
  explicit MatMulPrepacked(const OpKernelInfo& info);

  ~MatMulPrepacked();

  Status Compute(OpKernelContext* context) const override;
 private:
  std::unique_ptr<MLAS_GEMM_PARAMETERS> gemm_params_;
};
}  // namespace onnxruntime

