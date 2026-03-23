// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_node_proto_helper.h"
#include "deform_conv_attributes.h"
#include <memory>
#include <mutex>

namespace onnxruntime {

template <typename T>
struct DeformConvKernelMetaCacheData;

template <typename T>
class DeformConv : public OpKernel {
 public:
  explicit DeformConv(const OpKernelInfo& info) : OpKernel(info), attrs_(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  DeformConvAttributes attrs_;
  mutable std::mutex kernel_meta_cache_mu_;
  mutable std::shared_ptr<const DeformConvKernelMetaCacheData<T>> kernel_meta_cache_;
};

}  // namespace onnxruntime
