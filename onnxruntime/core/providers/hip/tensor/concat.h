// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/hip/hip_common.h"
#include "core/providers/cpu/tensor/concat.h"

namespace onnxruntime {
namespace hip {

class Concat final : public HipKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : HipKernel(info), ConcatBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace hip
}  // namespace onnxruntime
