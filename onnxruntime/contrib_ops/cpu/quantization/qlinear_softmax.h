// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class QLinearSoftmax final : public OpKernel {
 public:
  QLinearSoftmax(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  gsl::span<const uint32_t> GetLookupTable(OpKernelContext* context, gsl::span<uint32_t> lookup_table_span, size_t reduce_len) const;

  Status ComputeInternal(OpKernelContext* context, const Tensor& input, Tensor& output, gsl::span<const uint32_t> lookup_table, int axis, concurrency::ThreadPool* thread_pool) const;

  Status ComputeImplOpset13(OpKernelContext* context, const Tensor& input, Tensor& output,
                            gsl::span<const uint32_t> lookup_table, concurrency::ThreadPool* thread_pool) const;

 private:
  std::vector<uint32_t> fixed_lookup_table_;
  int axis_ = -1;
  int opset_ = 1;
  bool is_signed_{false};
};

}  // namespace contrib
}  // namespace onnxruntime
