// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class QLinearSoftmax final : public OpKernel {
 public:
  using EXP_OUT_DTYPE = float;  // or uint32_t if uint32_t is preferred.
  QLinearSoftmax(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  gsl::span<const EXP_OUT_DTYPE> GetLookupTable(
      OpKernelContext* context,
      gsl::span<EXP_OUT_DTYPE> lookup_table_span, size_t reduce_len) const;

  Status ComputeInternal(OpKernelContext* context, const Tensor& input,
                         Tensor& output, gsl::span<const EXP_OUT_DTYPE> lookup_table,
                         int axis, concurrency::ThreadPool* thread_pool) const;

  Status ComputeImplOpset13(OpKernelContext* context, const Tensor& input, Tensor& output,
                            gsl::span<const EXP_OUT_DTYPE> lookup_table,
                            concurrency::ThreadPool* thread_pool) const;

 private:
  std::vector<EXP_OUT_DTYPE> fixed_lookup_table_;
  int axis_ = -1;
  int opset_ = 1;
  bool is_signed_{false};
};

}  // namespace contrib
}  // namespace onnxruntime
