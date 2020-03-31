// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T, typename U>
class Scaler final : public CudaKernel {
 public:
  Scaler(const OpKernelInfo& info):CudaKernel(info){}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
