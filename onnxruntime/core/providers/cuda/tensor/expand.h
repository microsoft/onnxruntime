// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class Expand final : public CudaKernel {
 public:
  Expand(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

Status ComputeOutputShape(
    const std::string& node_name,
    const TensorShape& lhs_shape,
    const TensorShape& rhs_shape,
    TensorShape& out_shape);

}  // namespace cuda
}  // namespace onnxruntime
