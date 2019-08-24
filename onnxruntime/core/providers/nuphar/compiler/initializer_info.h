// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/tensor.h"
#include <tvm/tvm.h>

// TODO: move to nuphar
namespace onnxruntime {
namespace nuphar {

// TODO: move it to weight layout place
struct WeightLayoutCodegenInfo {
  const Tensor* marshalled_initializer = nullptr;  // TODO: change it to unique_ptr
  std::string layout = "";                         // layout name
  tvm::Tensor marshalled_tensor;
  tvm::Tensor unmarshalled_tensor;
  bool is_marshalled;

  WeightLayoutCodegenInfo(const tvm::Tensor& tvm_tensor)
      : marshalled_tensor(tvm_tensor), unmarshalled_tensor(tvm_tensor), is_marshalled(false) {}
};

struct InitializerInfo {
  const Tensor* original_initializer = nullptr;  // original ort tensor
  std::unique_ptr<WeightLayoutCodegenInfo> layout_info = nullptr;

  InitializerInfo(const Tensor* tensor) : original_initializer(tensor) {}
};

using InitializerMap = std::map<std::string, InitializerInfo>;

}  // namespace nuphar
}  // namespace onnxruntime
