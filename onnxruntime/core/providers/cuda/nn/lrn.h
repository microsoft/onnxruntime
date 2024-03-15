// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

class CudnnLRNDescriptor final {
 public:
  CudnnLRNDescriptor();
  ~CudnnLRNDescriptor();
  Status Set(uint32_t N, double alpha, double beta, double K);
  operator cudnnLRNDescriptor_t() const { return desc_; }

 private:
  cudnnLRNDescriptor_t desc_;
};

template <typename T, bool Layout>
class LRN : public CudaKernel {
 public:
  LRN(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;

 private:
  CudnnLRNDescriptor norm_desc_;
};

}  // namespace cuda
}  // namespace onnxruntime
