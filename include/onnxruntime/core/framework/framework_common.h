// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "run_options.h"

struct OrtCustomOpDomain {
  std::string domain_;
  std::vector<const OrtCustomOp*> custom_ops_;
};

namespace onnxruntime {  // forward declarations
class Model;
class GraphTransformer;
class NodeArg;
}  // namespace onnxruntime

namespace onnxruntime {
using InputDefList = std::vector<const onnxruntime::NodeArg*>;
using OutputDefList = std::vector<const onnxruntime::NodeArg*>;

using NameMLValMap = std::unordered_map<std::string, OrtValue>;
}  // namespace onnxruntime
