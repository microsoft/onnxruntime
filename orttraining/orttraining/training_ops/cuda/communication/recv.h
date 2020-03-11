// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_HOROVOD

#pragma once
#include <mpi.h>
#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Recv final : public CudaKernel {
public:
  Recv(const OpKernelInfo& info) : CudaKernel(info) {
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

}  // namespace cuda
}  // namespace onnxruntime

#endif