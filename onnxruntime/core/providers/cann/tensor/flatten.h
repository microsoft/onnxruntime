// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/cann_kernel.h"

namespace onnxruntime {
namespace cann {

template <typename T>
class Flatten final : public CannKernel {
 public:
  Flatten(const OpKernelInfo& info) : CannKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;
  Status Prepare(OpKernelContext* ctx, CannPreparation& prepare) const;

 private:
  int64_t axis_;
};

}  // namespace cann
}  // namespace onnxruntime
