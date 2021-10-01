// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/selectors_actions/nnapi_qdq_selector_helper.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

namespace onnxruntime {

//
// Selections
//

ConstNodesToOptimize::ConstNodesToOptimize(std::vector<const Node*>& input_nodes,
                                           const Node& target_node,
                                           std::vector<const Node*>& output_nodes,
                                           int num_input_defs, int num_output_defs)
    : num_inputs{num_input_defs == -1 ? gsl::narrow_cast<int>(input_nodes.size()) : num_input_defs},
      num_outputs{num_output_defs == -1 ? gsl::narrow_cast<int>(output_nodes.size()) : num_output_defs} {
  if (num_input_defs != -1) {
    variadic_input_ = true;
    num_variadic_inputs_ = gsl::narrow_cast<int>(input_nodes.size()) - num_input_defs + 1;
  }

  if (num_output_defs != -1) {
    variadic_output_ = true;
    num_variadic_outputs_ = gsl::narrow_cast<int>(output_nodes.size()) - num_output_defs + 1;
  }

  nodes_.reserve(NumInputEntries() + 1 + NumOutputEntries());
  std::copy(input_nodes.begin(), input_nodes.end(), std::back_inserter(nodes_));
  nodes_.push_back(&target_node);
  std::copy(output_nodes.begin(), output_nodes.end(), std::back_inserter(nodes_));
}

std::vector<Node*> NodesToOptimize::Inputs(const std::vector<int>& indexes, bool required) const {
  std::vector<Node*> results;
  results.reserve(NumInputEntries());

  for (auto idx : indexes) {
    if (idx == num_inputs - 1 && HasVariadicInput()) {
      for (int i = 0, end = NumVariadicInputs(); i < end; ++i) {
        results.push_back(GetNode(idx + i, required));
      }
    } else {
      results.push_back(GetNode(idx, required));
    }
  }

  return results;
}

std::vector<Node*> NodesToOptimize::Outputs(const std::vector<int>& indexes, bool required) const {
  std::vector<Node*> results;
  results.reserve(NumOutputEntries());

  // offset by all the inputs and the target node
  const int offset = NumInputEntries() + 1;

  for (auto idx : indexes) {
    if (idx == num_outputs - 1 && HasVariadicOutput()) {
      for (int i = 0, end = NumVariadicOutputs(); i < end; ++i) {
        results.push_back(GetNode(offset + idx + i, required));
      }
    } else {
      results.push_back(GetNode(offset + idx, required));
    }
  }

  return results;
}

std::vector<Node*> NodesToOptimize::GetNodesAtLocation(const NodeLocation& location, bool required) const {
  if (location.type == NodeType::kInput) {
    return Inputs({location.index}, required);
  } else if (location.type == NodeType::kOutput) {
    return Outputs({location.index}, required);
  } else
    return {&Target()};
};

}  // namespace onnxruntime
