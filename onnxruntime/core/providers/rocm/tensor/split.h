// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/hip/hip_common.h"
#include "core/providers/cpu/tensor/split.h"

namespace onnxruntime {
namespace rocm {

class Split final : public RocmKernel, public SplitBase {
 public:
  Split(const OpKernelInfo& info) : RocmKernel(info), SplitBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
