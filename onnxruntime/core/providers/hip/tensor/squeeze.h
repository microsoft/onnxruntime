// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/hip/hip_common.h"
#include "core/providers/cpu/tensor/squeeze.h"

namespace onnxruntime {
namespace hip {

class Squeeze final : public SqueezeBase, public HipKernel {
 public:
  Squeeze(const OpKernelInfo& info) : SqueezeBase(info), HipKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace hip
}  // namespace onnxruntime
