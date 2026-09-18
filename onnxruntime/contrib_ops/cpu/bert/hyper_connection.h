// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime::contrib {

template <typename T>
class BranchwiseRMSNorm final : public OpKernel {
 public:
  explicit BranchwiseRMSNorm(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  float epsilon_;
  int64_t num_branches_;
};

template <typename T>
class ScaledSiLU final : public OpKernel {
 public:
  explicit ScaledSiLU(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  float alpha_;
};

template <typename T>
class HyperConnectionPreMix final : public OpKernel {
 public:
  explicit HyperConnectionPreMix(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t num_branches_;
  float reduction_scale_;
};

template <typename T>
class HyperConnectionPostMix final : public OpKernel {
 public:
  explicit HyperConnectionPostMix(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t num_branches_;
};

}  // namespace onnxruntime::contrib
