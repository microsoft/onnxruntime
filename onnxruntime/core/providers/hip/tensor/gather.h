// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/hip/hip_common.h"
#include "core/providers/cpu/tensor/gather.h"

namespace onnxruntime {
namespace rocm {

class Gather final : public RocmKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : RocmKernel(info), GatherBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
