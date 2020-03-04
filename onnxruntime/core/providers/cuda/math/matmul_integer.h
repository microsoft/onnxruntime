// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {
template <typename T1, typename T2>
class MatMulInteger final : public CudaKernel {
  using Base = CudaKernel;

 public:
  MatMulInteger(const OpKernelInfo& info) : CudaKernel(info) {
    has_a_zero_point_ = false;
    has_b_zero_point_ = false;
    if (info.GetInputCount() > 2) {
      has_a_zero_point_ = true;
    }
    if (info.GetInputCount() > 3) {
      has_b_zero_point_ = true;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  // pad matrix and B to make their leading dimension be multiples of *align_size*
  Status PadMatrix(
      int64_t row,
      int64_t col,
      int64_t align_size,
      const int8_t*& src,
      int64_t& pad_size,
      IAllocatorUniquePtr<int8_t>& temp_mem_holder) const;

 private:
  bool has_a_zero_point_;
  bool has_b_zero_point_;
};

}  // namespace cuda
}  // namespace onnxruntime
