// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include <vector>
#include "core/graph/function.h"
//TODO: we need to make it a stand-alone header because both graph.cc and model.cc need to implement create instance of the graph object.
//Right now only functions_ has issue because it use vector of unique-ptr, maybe we should extend this to GraphImpl later.
namespace onnxruntime {
struct FunctionContainer {
  std::vector<std::unique_ptr<::onnxruntime::Function>> functions_;
};
}  // namespace onnxruntime
