// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class QLinearLeakyRelu final : public OpKernel {
 public:
  QLinearLeakyRelu(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  const float alpha_;
  bool is_fixed_parameters_;   // Fixed Scale and Zero Point for both x and y
  uint8_t fixed_lookup_table_[256];  // when is const paramter, table value is here.
};

}  // namespace contrib
}  // namespace onnxruntime
