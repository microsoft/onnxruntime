// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "contrib_ops/cpu/bert/attention.h"
#include "core/providers/cuda/math/cublas_gemm_algo_selector.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class Attention final : public CudaKernel, public AttentionBase {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  CublasGemmAlgoSelector gemm_algo_selector;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
