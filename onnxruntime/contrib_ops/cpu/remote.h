// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class RemoteCall : public OpKernel {
 public:
  RemoteCall(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<std::string>("uri", uri_);
    info.GetAttrOrDefault<std::string>("key", key_);
  };

  common::Status Compute(OpKernelContext* context) const override;

 private:
  std::string uri_;
  std::string key_;
};

}  // namespace contrib
}  // namespace onnxruntime