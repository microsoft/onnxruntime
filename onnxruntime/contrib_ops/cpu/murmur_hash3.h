// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class MurmurHash3 final : public OpKernel {
 public:
  MurmurHash3(const OpKernelInfo& info) : OpKernel(info) {
    seed_ = static_cast<uint32_t>(info.GetAttrOrDefault<int64_t>("seed", 0));
    is_positive_ = info.GetAttrOrDefault<int64_t>("positive", 1) == 1;
  }

  Status Compute(OpKernelContext* context) const override;

private:
  void MurmurHash3_x86_32(const void* key, int len, uint32_t seed, void* out) const;

private :
  uint32_t seed_;
  bool is_positive_{true};
};
}  // namespace contrib
}  // namespace onnxruntime
