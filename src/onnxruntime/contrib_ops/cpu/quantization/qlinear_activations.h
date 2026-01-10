// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class QLinearLookupBase : public OpKernel {
 public:
  QLinearLookupBase(const OpKernelInfo& info)
      : OpKernel(info), fixed_lookup_table_() {
  }

  //  protected:
  template <typename Transformer>
  Status ComputeBase(OpKernelContext* context, Transformer fn) const;

  // Should be called in derived class's constructor
  template <typename Transformer>
  void BuildLookupTableIfFixed(const OpKernelInfo& info, Transformer fn);

  // when input quantizaton parameters are const, pre-compute table value.
  // After construction, non-zero size means pre-computed. Save space when not pre-computed.
  std::vector<uint8_t> fixed_lookup_table_;
};

template <typename T>
class QLinearLeakyRelu final : public QLinearLookupBase<T> {
 public:
  QLinearLeakyRelu(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  const float alpha_;
};

template <typename T>
class QLinearSigmoid final : public QLinearLookupBase<T> {
 public:
  QLinearSigmoid(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
