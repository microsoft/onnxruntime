// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/armnn/armnn_execution_provider.h"
#include "core/providers/acl/nhwc/nhwc_ops.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep {

template <typename T>
class ReorderInput : public ::onnxruntime::acl::ReorderInput<T> {
 public:
  ReorderInput(const OpKernelInfo& info) : onnxruntime::acl::ReorderInput<T>(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReorderOutput : public ::onnxruntime::acl::ReorderOutput<T> {
 public:
  ReorderOutput(const OpKernelInfo& info) : onnxruntime::acl::ReorderOutput<T>(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t channels_;
};

}  // namespace armnn_ep
}  // namespace onnxruntime
