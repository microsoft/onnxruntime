// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class Compress final : public CudaKernel {
 public:
  Compress(const OpKernelInfo& info) : CudaKernel(info) {
    has_axis_ = info.GetAttr("axis", &axis_).IsOK();
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  bool has_axis_;
};

}  // namespace cuda
}  // namespace onnxruntime
