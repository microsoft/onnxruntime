// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/hip/hip_common.h"
#include "core/providers/cpu/tensor/split.h"
#include "orttraining/training_ops/cpu/tensor/split.h"

namespace onnxruntime {
namespace hip {

class SplitTraining final : public HipKernel, public SplitBase {
 public:
  SplitTraining(const OpKernelInfo& info) : HipKernel(info), SplitBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace hip
}  // namespace onnxruntime
