// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "tensorize_base.h"

namespace onnxruntime {
namespace tvm_codegen {

class Gemv16bitTensorization : public TensorizeBase {
 public:
  Gemv16bitTensorization(const std::string& name, const std::vector<int32_t>& vshape);

  virtual ~Gemv16bitTensorization() = default;

  tvm::TensorIntrin CreateTensorIntrin() override;
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
