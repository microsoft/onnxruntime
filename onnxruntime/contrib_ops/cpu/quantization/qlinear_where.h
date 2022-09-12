// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "qlinear_util.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/tensor/where_op.h"

namespace onnxruntime {
namespace contrib {

class QLinearWhere final : public OpKernel {
 public:
  QLinearWhere(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  const size_t expected_input_count = 9;
  std::vector<uint8_t> y_fixed_lookup_table_;
  std::vector<uint8_t> x_fixed_lookup_table_;
  int y_fixed_table_attr_ = 0;
  int x_fixed_table_attr_ = 0;
};

}  // namespace contrib
}  // namespace onnxruntime
