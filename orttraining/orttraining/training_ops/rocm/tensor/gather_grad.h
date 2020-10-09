// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

class GatherGrad final : public RocmKernel {
 public:
  GatherGrad(const OpKernelInfo& info) : RocmKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(), "Missing/Invalid 'axis' attribute value");
  }
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

}  // namespace rocm
}  // namespace onnxruntime
