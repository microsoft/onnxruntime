// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "contrib_ops/cpu/bert/attention.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class AttentionDynamicQuant final : public CudaKernel, public AttentionBase {
 using Base = CudaKernel;
 public:
  AttentionDynamicQuant(const OpKernelInfo& info);
  ~AttentionDynamicQuant();

  Status PadMatrix(
      int row,
      int col,
      int align_size,
      const int8_t*& src,
      int& pad_size,
      IAllocatorUniquePtr<int8_t>& temp_mem_holder) const;

  Status ComputeInternal(OpKernelContext* context) const override;
  
  private:
  void CacheWeights(const OpKernelInfo& info);
  bool cache_weight;
  cublasLtMatrixLayout_t a_desc;
  cublasLtMatrixLayout_t AtransformDesc;
  IAllocatorUniquePtr<int8_t> a_transform;
  cublasLtMatrixTransformDesc_t transform_desc;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
