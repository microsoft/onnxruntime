// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/bias_skip_layer_norm_fusion.h"

#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {

/**
Skip Layer Normalization with bias will fuse Add(MatMul, bias) + SkipLayerNormalization into one node.

Before fusion:
    MatMul         [skip]
      |              |
   Add(bias)         |
           \         |
        SkipLayerNormalization (4 inputs: input, skip, gamma, beta)

After fusion:
    MatMul    [skip]
        \       /
    SkipLayerNormalization (5 inputs: input, skip, gamma, beta, bias)

Note: Also handles a Cast between MatMul and Add (for fp16 models):
    MatMul → Cast → Add(bias) → SkipLayerNormalization
*/

Status BiasSkipLayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                          const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;

  auto get_bias_info = [&](Graph& g, NodeArg& bias_arg, bool& is_1d_bias, int64_t& bias_hidden_size) {
    is_1d_bias = false;
    bias_hidden_size = -1;

    const TensorShapeProto* bias_shape = bias_arg.Shape();
    if (bias_shape != nullptr) {
      is_1d_bias = (bias_shape->dim_size() == 1);
      if (is_1d_bias) {
        const auto& dim0 = bias_shape->dim(0);
        if (dim0.has_dim_value()) {
          bias_hidden_size = dim0.dim_value();
        }
      }
    } else {
      // For constant initializers from an outer scope, NodeArg::Shape() may be null.
      // Fall back to checking the TensorProto dims to confirm that the bias is 1D.
      const TensorProto* bias_initializer =
          graph_utils::GetConstantInitializer(g, bias_arg.Name(), true);
      if (bias_initializer != nullptr) {
        is_1d_bias = (bias_initializer->dims_size() == 1);
        if (is_1d_bias && bias_initializer->dims_size() == 1) {
          bias_hidden_size = bias_initializer->dims(0);
        }
      }
    }
  };

  for (auto node_index : node_topology_list) {
    Node* p_sln = graph.GetNode(node_index);
    if (p_sln == nullptr) continue;  // node was removed in an earlier fusion

    Node& sln_node = *p_sln;
    ORT_RETURN_IF_ERROR(Recurse(sln_node, modified, graph_level, logger));

    // Must be a SkipLayerNormalization node in the Microsoft custom domain.
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sln_node, "SkipLayerNormalization", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(sln_node, GetCompatibleExecutionProviders())) {
      continue;
    }

    // Must have exactly 4 inputs (input, skip, gamma, beta) – bias not yet absorbed.
    auto& sln_inputs = sln_node.MutableInputDefs();
    if (sln_inputs.size() != 4) {
      continue;
    }

    // Try each of the first two SLN inputs (input[0] = "input", input[1] = "skip") to find an Add
    // that adds a 1D constant bias to a MatMul result. Also consider a Cast between MatMul and Add
    // (common in fp16 models).
    Node* p_add = nullptr;
    int sln_add_input_index = -1;  // which SLN input (0 or 1) leads to the Add node
    int add_bias_index = -1;       // which Add input (0 or 1) is the 1D constant bias

    for (int sln_input_idx = 0; sln_input_idx <= 1 && p_add == nullptr; ++sln_input_idx) {
      for (int add_matmul_input_idx = 0; add_matmul_input_idx <= 1 && p_add == nullptr;
           ++add_matmul_input_idx) {
        // --- Path 1: SLN.input[sln_input_idx] ← Add ← MatMul (direct) ---
        std::vector<graph_utils::EdgeEndToMatch> path_matmul{
            {0, sln_input_idx, "Add", {7, 13, 14}, kOnnxDomain},
            {0, add_matmul_input_idx, "MatMul", {1, 9, 13}, kOnnxDomain}};

        std::vector<const Node::EdgeEnd*> edges;
        if (graph_utils::FindPath(sln_node, true, path_matmul, edges, logger)) {
          Node* candidate_add = const_cast<Node*>(&edges[0]->GetNode());

          if (candidate_add->GetExecutionProviderType() == sln_node.GetExecutionProviderType() &&
              candidate_add->GetOutputEdgesCount() == 1 &&
              !graph.NodeProducesGraphOutput(*candidate_add)) {
            int bias_idx = 1 - add_matmul_input_idx;
            NodeArg* bias_arg = candidate_add->MutableInputDefs()[bias_idx];

            if (graph_utils::NodeArgIsConstant(graph, *bias_arg)) {
              bool is_1d_bias = false;
              int64_t bias_hidden_size = -1;
              get_bias_info(graph, *bias_arg, is_1d_bias, bias_hidden_size);

              // If we know the bias is 1D, additionally check that its length matches the
              // hidden size expected by SkipLayerNormalization (gamma/beta) when that size
              // can be determined. If the sizes are known and incompatible, skip fusion.
              bool bias_matches_hidden = true;
              if (is_1d_bias) {
                int64_t sln_hidden_size = -1;

                auto get_sln_hidden_size_from_input = [&](size_t input_index) -> int64_t {
                  if (sln_node.InputDefs().size() <= input_index) {
                    return -1;
                  }
                  NodeArg* arg = sln_node.MutableInputDefs()[input_index];
                  if (arg == nullptr) {
                    return -1;
                  }

                  const TensorShapeProto* shape = arg->Shape();
                  if (shape != nullptr && shape->dim_size() == 1) {
                    const auto& dim0 = shape->dim(0);
                    if (dim0.has_dim_value()) {
                      return dim0.dim_value();
                    }
                  }

                  const TensorProto* initializer =
                      graph_utils::GetConstantInitializer(graph, arg->Name(), true);
                  if (initializer != nullptr && initializer->dims_size() == 1) {
                    return initializer->dims(0);
                  }

                  return -1;
                };

                // Try gamma (input index 2), then beta (input index 3) as the hidden-size source.
                sln_hidden_size = get_sln_hidden_size_from_input(2);
                if (sln_hidden_size == -1) {
                  sln_hidden_size = get_sln_hidden_size_from_input(3);
                }

                if (sln_hidden_size != -1 && bias_hidden_size != -1 && sln_hidden_size != bias_hidden_size) {
                  bias_matches_hidden = false;
                }
              }

              if (is_1d_bias && bias_matches_hidden) {
                p_add = candidate_add;
                sln_add_input_index = sln_input_idx;
                add_bias_index = bias_idx;
              }
            }
          }
        }

        if (p_add != nullptr) break;

        // --- Path 2: SLN.input[sln_input_idx] ← Add ← Cast ← MatMul (fp16 models) ---
        std::vector<graph_utils::EdgeEndToMatch> path_cast_matmul{
            {0, sln_input_idx, "Add", {7, 13, 14}, kOnnxDomain},
            {0, add_matmul_input_idx, "Cast", {1, 6, 9, 13, 15}, kOnnxDomain},
            {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain}};

        if (graph_utils::FindPath(sln_node, true, path_cast_matmul, edges, logger)) {
          Node* candidate_add = const_cast<Node*>(&edges[0]->GetNode());

          if (candidate_add->GetExecutionProviderType() == sln_node.GetExecutionProviderType() &&
              candidate_add->GetOutputEdgesCount() == 1 &&
              !graph.NodeProducesGraphOutput(*candidate_add)) {
            int bias_idx = 1 - add_matmul_input_idx;
            NodeArg* bias_arg = candidate_add->MutableInputDefs()[bias_idx];

            if (graph_utils::NodeArgIsConstant(graph, *bias_arg)) {
              bool is_1d_bias = false;
              int64_t bias_hidden_size = -1;
              const TensorShapeProto* bias_shape = bias_arg->Shape();
              if (bias_shape != nullptr) {
                is_1d_bias = (bias_shape->dim_size() == 1);
                if (is_1d_bias) {
                  const auto& dim0 = bias_shape->dim(0);
                  if (dim0.has_dim_value()) {
                    bias_hidden_size = dim0.dim_value();
                  }
                }
              } else {
                // For constant initializers from an outer scope, NodeArg::Shape() may be null.
                // Fall back to checking the TensorProto dims to confirm that the bias is 1D.
                const TensorProto* bias_initializer =
                    graph_utils::GetConstantInitializer(graph, bias_arg->Name(), true);
                if (bias_initializer != nullptr) {
                  is_1d_bias = (bias_initializer->dims_size() == 1);
                  if (is_1d_bias && bias_initializer->dims_size() == 1) {
                    bias_hidden_size = bias_initializer->dims(0);
                  }
                }
              }

              bool bias_matches_hidden = true;
              if (is_1d_bias) {
                // Derive the hidden size from SkipLayerNormalization's gamma input when available.
                const NodeArg* gamma_arg = sln_node.InputDefs()[2];
                int64_t sln_hidden_size = -1;
                const TensorShapeProto* gamma_shape = gamma_arg->Shape();
                if (gamma_shape != nullptr && gamma_shape->dim_size() == 1) {
                  const auto& dim0 = gamma_shape->dim(0);
                  if (dim0.has_dim_value()) {
                    sln_hidden_size = dim0.dim_value();
                  }
                } else {
                  const TensorProto* gamma_initializer =
                      graph_utils::GetConstantInitializer(graph, gamma_arg->Name(), true);
                  if (gamma_initializer != nullptr && gamma_initializer->dims_size() == 1) {
                    sln_hidden_size = gamma_initializer->dims(0);
                  }
                }

                if (bias_hidden_size > 0 && sln_hidden_size > 0 && bias_hidden_size != sln_hidden_size) {
                  bias_matches_hidden = false;
                }
              }

              if (is_1d_bias && bias_matches_hidden) {
                p_add = candidate_add;
                sln_add_input_index = sln_input_idx;
                add_bias_index = bias_idx;
              }
            }
          }
        }
      }
    }

    if (p_add == nullptr) continue;

    // Determine the non-bias Add input (MatMul / Cast output) and the SLN skip input.
    int add_non_bias_input_index = 1 - add_bias_index;
    int sln_skip_input_index = 1 - sln_add_input_index;

    // Build the new 5-input SkipLayerNormalization:
    //   input[0] = MatMul (or Cast) output   – the "input" tensor
    //   input[1] = skip                       – the "skip" tensor
    //   input[2] = gamma                      – unchanged
    //   input[3] = beta                       – unchanged
    //   input[4] = bias                       – absorbed from the Add node
    InlinedVector<NodeArg*> new_sln_inputs{
        p_add->MutableInputDefs()[add_non_bias_input_index],  // input (MatMul / Cast output)
        sln_inputs[sln_skip_input_index],                     // skip
        sln_inputs[2],                                        // gamma
        sln_inputs[3],                                        // beta
        p_add->MutableInputDefs()[add_bias_index]             // bias (1D constant)
    };

    Node& new_sln_node = graph.AddNode(
        graph.GenerateNodeName("SkipLayerNormalization"),
        "SkipLayerNormalization",
        "fused SkipLayerNormalization and bias Add",
        new_sln_inputs,
        sln_node.MutableOutputDefs(),
        {},
        kMSDomain);

    // Copy all attributes from the original SkipLayerNormalization node, ensuring epsilon is set.
    const NodeAttributes& sln_attrs = sln_node.GetAttributes();

    // First copy all non-epsilon attributes.
    for (const auto& attr_pair : sln_attrs) {
      if (attr_pair.first == "epsilon") {
        continue;
      }
      new_sln_node.AddAttributeProto(attr_pair.second);
    }

    // Then handle epsilon specifically so we can apply a default if it is missing.
    auto epsilon_it = sln_attrs.find("epsilon");
    if (epsilon_it != sln_attrs.end()) {
      new_sln_node.AddAttributeProto(epsilon_it->second);
    } else {
      new_sln_node.AddAttribute("epsilon", contrib::kDefaultSkipLayerNormEpsilon);
    }

    new_sln_node.SetExecutionProviderType(sln_node.GetExecutionProviderType());

    nodes_to_remove.push_back(*p_add);
    nodes_to_remove.push_back(sln_node);
    // Note: nodes are only actually removed after the full iteration (see below), so subsequent
    // iterations in this loop will still see these nodes but will not fuse them again because
    // (a) each node_index in node_topology_list is unique, and (b) the Add node's output edge
    // count is 1, preventing it from being matched a second time as a consumer of another SLN.
  }

  for (const auto& node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.get().Index());
  }

  if (!nodes_to_remove.empty()) {
    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
