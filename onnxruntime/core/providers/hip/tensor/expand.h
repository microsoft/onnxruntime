// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

class Expand final : public HipKernel {
 public:
  Expand(const OpKernelInfo& info) : HipKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

Status ComputeOutputShape(
    const std::string& node_name,
    const TensorShape& lhs_shape,
    const TensorShape& rhs_shape,
    TensorShape& out_shape);

}  // namespace hip
}  // namespace onnxruntime
