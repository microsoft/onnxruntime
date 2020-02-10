// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/hip/hip_common.h"
#include "core/providers/cpu/tensor/gather.h"

namespace onnxruntime {
namespace hip {

class Gather final : public HipKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : HipKernel(info), GatherBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace hip
}  // namespace onnxruntime
