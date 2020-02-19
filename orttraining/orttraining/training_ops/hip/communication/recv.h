// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_HOROVOD

#pragma once
#include <mpi.h>
#include "core/common/common.h"
#include "core/providers/hip/hip_common.h"
#include "common.h"

namespace onnxruntime {
namespace hip {

template <typename T>
class Recv final : public HipKernel {
public:
  Recv(const OpKernelInfo& info) : HipKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("tag", &tag_).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("src", &src_).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("dst", &dst_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

private:
  int64_t tag_;
  int64_t src_;
  int64_t dst_;
};

}  // namespace hip
}  // namespace onnxruntime

#endif