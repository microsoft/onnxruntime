// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class QLinearSoftmax final : public OpKernel {
 public:
  QLinearSoftmax(const OpKernelInfo& info);
  void BuildLookupTableIfFixed(const OpKernelInfo& info, size_t channels);
  Status Compute(OpKernelContext* context) const override;

 private:
  const uint32_t* GetLookupTable(OpKernelContext* context, size_t reduce_len) const;
  Status ComputeImpl(OpKernelContext* context, const Tensor& input, Tensor& output,
                     concurrency::ThreadPool* thread_pool, const uint32_t* lookup_table) const;

  Status ComputeImplOpset13(OpKernelContext* context, const Tensor& input, Tensor& output,
                            concurrency::ThreadPool* thread_pool, const uint32_t* lookup_table) const;

 private:
  std::vector<uint32_t> fixed_lookup_table_;
  mutable std::vector<uint32_t> tmp_lookup_table_;
  int axis_ = -1;
  int opset_ = 1;
};

}  // namespace contrib
}  // namespace onnxruntime
