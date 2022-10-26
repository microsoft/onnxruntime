// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

//#include "core/graph/graph_viewer.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

//class GraphViewer;

class CloudExecutionProvider : public IExecutionProvider {
 public:
  explicit CloudExecutionProvider(const std::unordered_map<std::string, std::string>& config);
  ~CloudExecutionProvider() = default;

 private:
  std::unordered_map<std::string, std::string> config_;
};

}  // namespace onnxruntime