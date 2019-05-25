// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "tensorize_base.h"

namespace onnxruntime {
namespace tvm_codegen {

class NaiveLLVMIRGemvTensorization : public TensorizeBase {
 public:
  NaiveLLVMIRGemvTensorization(const std::string& name);

  virtual ~NaiveLLVMIRGemvTensorization() = default;

  tvm::TensorIntrin CreateTensorIntrin() override;
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
