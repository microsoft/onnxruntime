// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

class SummaryScalarOp final : public OpKernel {
 public:
  explicit SummaryScalarOp(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext& context, const Tensor& input) const;

  std::vector<std::string> tags_;
};

class SummaryHistogramOp final : public OpKernel {
 public:
  explicit SummaryHistogramOp(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext& context, const Tensor& input) const;

  std::string tag_;
};

class SummaryMergeOp final : public OpKernel {
 public:
  explicit SummaryMergeOp(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;
};

class SummaryTextOp final : public OpKernel {
 public:
  explicit SummaryTextOp(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  std::string tag_;
};

}  // namespace contrib
}  // namespace onnxruntime
