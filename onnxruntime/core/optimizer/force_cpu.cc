// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/force_cpu.h"
#include "core/optimizer/utils.h"
#include <iostream>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status ForceCpu::Apply(Graph&, Node& node, RewriteRuleEffect&, const logging::Logger&) const {
  node.SetExecutionProviderType(kCpuExecutionProvider);
  const auto& next_node = *node.OutputNodesBegin();
  const_cast<Node&>(next_node).SetExecutionProviderType(kCpuExecutionProvider);
  const auto& cast_node = *next_node.OutputNodesBegin();
  const_cast<Node&>(cast_node).SetExecutionProviderType(kCpuExecutionProvider);
  return Status::OK();
}

bool ForceCpu::SatisfyCondition(const Graph&, const Node& node, const logging::Logger&) const {
  if (!graph_utils::IsSupportedProvider(node, {kJsExecutionProvider})) {
    return false;
  }
  if (*(node.InputDefs()[0]->Type()) != std::string("tensor(int64)")) {
    return false;
  }
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Unsqueeze", {1, 10, 11, 12}) ||
      (node.GetOutputEdgesCount() != 1 && node.GetInputEdgesCount() != 0)) {
    return false;
  }

  const auto& next_node = *node.OutputNodesBegin();
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Unsqueeze", {1, 10, 11, 12}) ||
      node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const auto& cast_node = *next_node.OutputNodesBegin();
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(cast_node, "Cast", {6, 8, 9, 12, 13, 18, 19}) ||
      node.GetOutputEdgesCount() != 1) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
