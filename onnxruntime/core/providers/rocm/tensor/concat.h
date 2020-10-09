// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/cpu/tensor/concat.h"

namespace onnxruntime {
namespace rocm {

class Concat final : public RocmKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : RocmKernel(info), ConcatBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
