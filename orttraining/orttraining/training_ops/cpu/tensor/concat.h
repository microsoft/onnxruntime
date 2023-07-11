// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/concat.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

class ConcatTraining : public OpKernel, public ConcatBase {
 public:
  ConcatTraining(const OpKernelInfo& info) : OpKernel(info), ConcatBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
