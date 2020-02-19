// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

class GatherGrad final : public HipKernel {
 public:
  GatherGrad(const OpKernelInfo& info) : HipKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(), "Missing/Invalid 'axis' attribute value");
  }
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

}  // namespace hip
}  // namespace onnxruntime
