// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class QuantizeWithOrder final : public CudaKernel {
 public:
  QuantizeWithOrder(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_input_;
  int order_output_;
};

class DequantizeWithOrder final : public CudaKernel {
 public:
  DequantizeWithOrder(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_input_;
  int order_output_;
  ONNX_NAMESPACE::TensorProto_DataType to_;
};

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000

cublasLtOrder_t GetCublasLtOrderAttr(const OpKernelInfo& info, const char* order_attr);

int64_t CalcLeadingDimensionLt(int64_t rows, int64_t cols, cublasLtOrder_t order);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
