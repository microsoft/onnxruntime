// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include <string>

namespace onnxruntime {
namespace contrib {

template <typename T>
class ShortConv final : public OpKernel {
 public:
  explicit ShortConv(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  std::string activation_;
  int64_t dilation_;
  float epsilon_;
};

template <typename T>
class NgramHashMapping final : public OpKernel {
 public:
  explicit NgramHashMapping(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t max_ngram_size_;
  int64_t n_head_per_ngram_;
  T pad_id_;
};

template <typename T>
class EngramGate final : public OpKernel {
 public:
  explicit EngramGate(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  float epsilon_;
};

}  // namespace contrib
}  // namespace onnxruntime
