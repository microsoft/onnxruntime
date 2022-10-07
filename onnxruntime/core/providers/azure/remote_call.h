// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace azure {

class RemoteCall : public OpKernel {
 public:
  RemoteCall(const OpKernelInfo& info) : OpKernel(info) {
    for (uint32_t i = 0; i < info.GetInputCount(); ++i) {
      const auto* input = info.node().InputDefs().at(i);
      input_names_.push_back(input->Name());
    }
    for (uint32_t i = 0; i < info.GetOutputCount(); ++i) {
      const auto* output = info.node().OutputDefs().at(i);
      output_names_.push_back(output->Name());
    }
  }
  common::Status Compute(OpKernelContext* context) const override;

 private:
  onnxruntime::InlinedVector<std::string> input_names_;
  onnxruntime::InlinedVector<std::string> output_names_;
};

}  // namespace azure
}  // namespace onnxruntime