// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_HOROVOD

#pragma once
#include "core/common/common.h"
#include "core/providers/hip/hip_common.h"


namespace onnxruntime {
namespace hip {

class Send final : public HipKernel {
public:
  Send(const OpKernelInfo& info) : HipKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("tag", &tag_).IsOK());
    ORT_ENFORCE(info.GetAttrs<int64_t>("element_types", element_types_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

private:
  int64_t tag_;
  std::vector<int64_t> element_types_;
};

}  // namespace hip
}  // namespace onnxruntime

#endif