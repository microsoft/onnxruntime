// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "tensorize_base.h"

namespace onnxruntime {
namespace nuphar {

class NaiveLLVMIRGemvTensorization : public tvm_codegen::TensorizeBase {
 public:
  NaiveLLVMIRGemvTensorization(const std::string& name);

  virtual ~NaiveLLVMIRGemvTensorization() = default;

  tvm::TensorIntrin CreateTensorIntrin() override;
};

}  // namespace nuphar
}  // namespace onnxruntime
