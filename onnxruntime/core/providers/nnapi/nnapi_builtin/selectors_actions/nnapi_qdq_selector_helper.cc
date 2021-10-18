// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/selectors_actions/nnapi_qdq_selector_helper.h"
#include "gsl/gsl"

namespace onnxruntime {

//
// Selections
//

inline ConstNodesToOptimize::ConstNodesToOptimize(std::vector<const Node*>& input_nodes,
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

}  // namespace onnxruntime
