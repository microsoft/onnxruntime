// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/tensorize_base.h"

namespace onnxruntime {
namespace nuphar {

class TensorizeIntGemv16bit : public tvm_codegen::TensorizeBase {
 public:
  TensorizeIntGemv16bit(const std::string& name, const std::vector<int32_t>& vshape);

  virtual ~TensorizeIntGemv16bit() = default;

  tvm::TensorIntrin CreateTensorIntrin() override;
};

}  // namespace nuphar
}  // namespace onnxruntime
