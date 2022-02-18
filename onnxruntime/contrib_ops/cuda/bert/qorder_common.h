// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(USE_CUDA) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000

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
  cublasLtOrder_t order_input_;
  cublasLtOrder_t order_output_;
};

class DequantizeWithOrder final : public CudaKernel {
 public:
  DequantizeWithOrder(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  cublasLtOrder_t order_input_;
  cublasLtOrder_t order_output_;
  ONNX_NAMESPACE::TensorProto_DataType to_;
};

cublasLtOrder_t GetCublasLtOrderAttr(const OpKernelInfo& info, const char* order_attr);

int64_t CalcLeadingDimensionLt(int64_t rows, int64_t cols, cublasLtOrder_t order);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif
