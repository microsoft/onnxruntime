// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

class Expand final : public RocmKernel {
 public:
  Expand(const OpKernelInfo& info) : RocmKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

Status ComputeOutputShape(
    const std::string& node_name,
    const TensorShape& lhs_shape,
    const TensorShape& rhs_shape,
    TensorShape& out_shape);

}  // namespace rocm
}  // namespace onnxruntime
