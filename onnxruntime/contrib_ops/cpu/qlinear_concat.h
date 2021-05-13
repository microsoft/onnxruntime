// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/tensor/concat.h"

namespace onnxruntime {
namespace contrib {

class QLinearConcat final : public OpKernel, public ConcatBase {
 public:
  QLinearConcat(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<std::vector<uint8_t>> fixed_lookup_tables_;
};

}  // namespace contrib
}  // namespace onnxruntime
