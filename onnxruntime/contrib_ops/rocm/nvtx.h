// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/rocm_common.h"

using namespace onnxruntime::rocm;

namespace onnxruntime {
namespace contrib {
namespace rocm {

class NvtxPush final : public RocmKernel {
 public:
  NvtxPush(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

class NvtxPop final : public RocmKernel {
 public:
  NvtxPop(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};


}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
