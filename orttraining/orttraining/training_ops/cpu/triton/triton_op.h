// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if ENABLE_TRITONOP

#pragma once

#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {
namespace contrib {

class TritonOp final : public OpKernel {
 public:
  TritonOp(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("func_name", &func_name_));
    ORT_THROW_IF_ERROR(info.GetAttr("onnx_key", &onnx_key_));
    ORT_THROW_IF_ERROR(info.GetAttr("onnx_string", &onnx_string_));
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool IsBoolOutput(size_t index) const;

  std::string func_name_;
  int64_t onnx_key_;
  std::string onnx_string_;
};

}  // namespace contrib
}  // namespace onnxruntime

#endif  // ENABLE_TRITONOP
