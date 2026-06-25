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
        if (is_1d_bias) {
          bias_hidden_size = bias_initializer->dims(0);
        }
      }
    }
  };

  // Helper: derive the hidden size from a single SLN 1-D input (gamma or beta).
  // Returns -1 when the size cannot be determined.
  auto get_sln_hidden_size_from_input = [&](const Node& sln, size_t input_index) -> int64_t {
    if (sln.InputDefs().size() <= input_index) {
      return -1;
    }
    const NodeArg* arg = sln.InputDefs()[input_index];
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

  // Helper: derive the SLN hidden size by trying gamma (input 2) then beta (input 3).
  auto get_sln_hidden_size = [&](const Node& sln) -> int64_t {
    int64_t size = get_sln_hidden_size_from_input(sln, 2);
    if (size == -1) {
      size = get_sln_hidden_size_from_input(sln, 3);
    }
    return size;
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

    // Helper: validate a candidate Add node and, if it is a compatible bias-add, accept it
    // by setting p_add/sln_add_input_index/add_bias_index. Returns true on acceptance.
    // Both Path 1 and Path 2 call this so all acceptance criteria stay in one place.
    auto try_accept_add = [&](Node* candidate_add, int add_matmul_input_idx, int sln_input_idx) -> bool {
      if (candidate_add->GetExecutionProviderType() != sln_node.GetExecutionProviderType() ||
          candidate_add->GetOutputEdgesCount() != 1 ||
          graph.NodeProducesGraphOutput(*candidate_add)) {
        return false;
      }
      int bias_idx = 1 - add_matmul_input_idx;
      NodeArg* bias_arg = candidate_add->MutableInputDefs()[bias_idx];
      if (!graph_utils::NodeArgIsConstant(graph, *bias_arg)) {
        return false;
      }
      bool is_1d_bias = false;
      int64_t bias_hidden_size = -1;
      get_bias_info(graph, *bias_arg, is_1d_bias, bias_hidden_size);
      if (!is_1d_bias) {
        return false;
      }
      // Verify the bias length matches the SLN hidden size.
      // Try gamma/beta first; if not available, fall back to the last dimension of the Add's
      // non-bias input (the MatMul/Cast output, whose last dim equals the hidden size).
      int64_t sln_hidden_size = get_sln_hidden_size(sln_node);
      if (sln_hidden_size == -1) {
        const NodeArg* non_bias_arg = candidate_add->MutableInputDefs()[add_matmul_input_idx];
        const TensorShapeProto* non_bias_shape = non_bias_arg->Shape();
        if (non_bias_shape != nullptr && non_bias_shape->dim_size() > 0) {
          const auto& last_dim = non_bias_shape->dim(non_bias_shape->dim_size() - 1);
          if (last_dim.has_dim_value()) {
            sln_hidden_size = last_dim.dim_value();
          }
        }
      }
      // Require positive proof that bias length == hidden size; bail if either is still unknown.
      if (sln_hidden_size == -1 || bias_hidden_size == -1 || sln_hidden_size != bias_hidden_size) {
        return false;
      }
      p_add = candidate_add;
      sln_add_input_index = sln_input_idx;
      add_bias_index = bias_idx;
      return true;
    };

    for (int sln_input_idx = 0; sln_input_idx <= 1 && p_add == nullptr; ++sln_input_idx) {
      for (int add_matmul_input_idx = 0; add_matmul_input_idx <= 1 && p_add == nullptr;
           ++add_matmul_input_idx) {
        // --- Path 1: SLN.input[sln_input_idx] ← Add ← MatMul (direct) ---
        std::vector<graph_utils::EdgeEndToMatch> path_matmul{
            {0, sln_input_idx, "Add", {7, 13, 14}, kOnnxDomain},
            {0, add_matmul_input_idx, "MatMul", {1, 9, 13}, kOnnxDomain}};

        std::vector<const Node::EdgeEnd*> edges;
        if (graph_utils::FindPath(sln_node, true, path_matmul, edges, logger)) {
          try_accept_add(const_cast<Node*>(&edges[0]->GetNode()), add_matmul_input_idx, sln_input_idx);
        }

        if (p_add != nullptr) break;

        // --- Path 2: SLN.input[sln_input_idx] ← Add ← Cast ← MatMul (fp16 models) ---
        std::vector<graph_utils::EdgeEndToMatch> path_cast_matmul{
            {0, sln_input_idx, "Add", {7, 13, 14}, kOnnxDomain},
            {0, add_matmul_input_idx, "Cast", {1, 6, 9, 13, 15}, kOnnxDomain},
            {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain}};

        if (graph_utils::FindPath(sln_node, true, path_cast_matmul, edges, logger)) {
          try_accept_add(const_cast<Node*>(&edges[0]->GetNode()), add_matmul_input_idx, sln_input_idx);
        }
      }
    }

    if (p_add == nullptr) continue;

    // Determine the non-bias Add input (MatMul / Cast output).
    int add_non_bias_input_index = 1 - add_bias_index;

    // Snapshot all information we need from the original nodes before modifying the graph.
    // Build the new 5-input SkipLayerNormalization by replacing only the SLN input slot that
    // was fed by the bias-Add with the Add's non-bias (MatMul/Cast) input.  All other SLN inputs
    // stay in their original positions.  Preserving the input[0]/input[1] order is important
    // because SkipLayerNormalization derives its output shape from input[0] while input[1]
    // supports broadcasting; swapping them would silently change semantics.
    InlinedVector<NodeArg*> new_sln_inputs{
        sln_inputs[0],                             // original input[0] (replaced below if needed)
        sln_inputs[1],                             // original input[1] (replaced below if needed)
        sln_inputs[2],                             // gamma – unchanged
        sln_inputs[3],                             // beta – unchanged
        p_add->MutableInputDefs()[add_bias_index]  // bias (1D constant) – absorbed from Add
    };
    // Replace only the SLN slot that was connected to the bias-Add.
    new_sln_inputs[sln_add_input_index] = p_add->MutableInputDefs()[add_non_bias_input_index];

    // Snapshot the outputs of the original SkipLayerNormalization so we can safely remove it
    // before creating the replacement node while preserving the same graph outputs.
    InlinedVector<NodeArg*> new_sln_outputs;
    {
      auto& sln_output_defs = sln_node.MutableOutputDefs();
      new_sln_outputs.assign(sln_output_defs.begin(), sln_output_defs.end());
    }

    // Snapshot attributes and execution provider type from the original SLN node.
    const NodeAttributes sln_attrs = sln_node.GetAttributes();
    const std::string sln_ep = sln_node.GetExecutionProviderType();

    // Capture outgoing edges from the original SLN node BEFORE removing any nodes.
    // RemoveNodeOutputEdges clears the edge list, so this must precede removal to
    // ensure downstream consumers are correctly rewired to the new fused node.
    std::vector<std::tuple<NodeIndex, int, int>> sln_output_edges;
    sln_output_edges.reserve(std::distance(sln_node.OutputEdgesBegin(), sln_node.OutputEdgesEnd()));
    for (auto it = sln_node.OutputEdgesBegin(); it != sln_node.OutputEdgesEnd(); ++it) {
      auto& edge = *it;
      sln_output_edges.emplace_back(edge.GetNode().Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
    }

    // Remove the original Add and SkipLayerNormalization nodes (and their output edges)
    // before adding the fused node to maintain the single-producer invariant for NodeArgs.
    graph_utils::RemoveNodeOutputEdges(graph, *p_add);
    graph.RemoveNode(p_add->Index());
    graph_utils::RemoveNodeOutputEdges(graph, sln_node);
    graph.RemoveNode(sln_node.Index());

    // The fused 5-input SkipLayerNormalization:
    //   input[0] = original SLN input[0] (unless the bias-Add was at SLN input[0])
    //   input[1] = original SLN input[1] (unless the bias-Add was at SLN input[1])
    //   input[2] = gamma                      – unchanged
    //   input[3] = beta                       – unchanged
    //   input[4] = bias                       – absorbed from the Add node
    Node& new_sln_node = graph.AddNode(
        graph.GenerateNodeName("SkipLayerNormalization"),
        "SkipLayerNormalization",
        "fused SkipLayerNormalization and bias Add",
        new_sln_inputs,
        new_sln_outputs,
        {},
        kMSDomain);

    // Copy all attributes from the original SkipLayerNormalization node, ensuring epsilon is set.

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

    new_sln_node.SetExecutionProviderType(sln_ep);

    // Rewire all downstream consumers from the original SLN node to the new fused node.
    for (const auto& edge_info : sln_output_edges) {
      graph.AddEdge(new_sln_node.Index(), std::get<0>(edge_info), std::get<1>(edge_info),
                    std::get<2>(edge_info));
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
