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
  NvtxPush(const OpKernelInfo& info) : RocmKernel(info) {
    info.GetAttr("label", &label_);
    info.GetAttr("cid", &correlationId_);
  }
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::string label_;
  int64_t correlationId_;
};

class NvtxPop final : public RocmKernel {
 public:
  NvtxPop(const OpKernelInfo& info) : RocmKernel(info) { 
    info.GetAttr("label", &label_);
    info.GetAttr("cid", &correlationId_);
  }
  Status ComputeInternal(OpKernelContext* context) const override;
 private:
  std::string label_;
  int64_t correlationId_;
};


}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
