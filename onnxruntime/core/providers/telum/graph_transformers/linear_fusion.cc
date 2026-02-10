// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telum_transformer_base.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {
namespace telum {

/**
 * @brief Fuses MatMul + Add patterns into Gemm for better performance
 *
 * Pattern: MatMul → Add becomes Gemm
 *
 * This is one of the most common patterns in transformer models:
 * - Q/K/V projections: Linear(x) = x @ W + b
 * - Output projections
 * - Feed-forward layers
 *
 * By fusing these operations, we can:
 * 1. Reduce memory traffic
 * 2. Leverage zDNN's optimized Gemm implementation
 * 3. Eliminate intermediate tensor allocations
 */
class LinearFusionTransformer : public TelumTransformerBase {
 public:
  LinearFusionTransformer()
      : TelumTransformerBase("LinearFusionTransformer",
                            {onnxruntime::kTelumExecutionProvider}) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                  const logging::Logger& logger) const override {
    GraphViewer graph_viewer(graph);
    const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

    for (auto node_index : node_topology_list) {
      auto* node = graph.GetNode(node_index);
      if (!node) continue;

      ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

      // Look for MatMul → Add pattern
      if (node->OpType() == "MatMul") {
        // Check if MatMul has static shapes
        if (!HasStaticShapes(*node)) {
          continue;
        }

        // Check if MatMul output is consumed by a single Add node
        const Node* add_node = GetSingleConsumer(*node);
        if (!add_node || add_node->OpType() != "Add") {
          continue;
        }

        // Check if Add has static shapes
        if (!HasStaticShapes(*add_node)) {
          continue;
        }

        // Verify the Add is adding a bias (not another matrix)
        if (!IsBiasAddition(*node, *add_node)) {
          continue;
        }

        // Perform fusion
        ORT_RETURN_IF_ERROR(FuseMatMulAdd(graph, *node, *add_node, modified, logger));
      }
    }

    return Status::OK();
  }

 private:
  /**
   * @brief Check if Add node is adding a bias to MatMul output
   */
  bool IsBiasAddition(const Node& matmul_node, const Node& add_node) const {
    // Get MatMul output shape
    auto matmul_output = matmul_node.OutputDefs()[0];
    auto matmul_shape = GetShape(matmul_output);

    if (matmul_shape.empty() || matmul_shape.size() != 2) {
      return false;
    }

    // Check both inputs of Add
    for (size_t i = 0; i < add_node.InputDefs().size(); ++i) {
      auto* input_def = add_node.InputDefs()[i];

      // Skip the MatMul output
      if (input_def == matmul_output) {
        continue;
      }

      // Check if the other input is a bias vector
      auto bias_shape = GetShape(input_def);

      // Bias should be 1D with size matching last dimension of MatMul output
      if (bias_shape.size() == 1 && bias_shape[0] == matmul_shape[1]) {
        return true;
      }

      // Or 2D with shape [1, N] or [M, N] where M=1
      if (bias_shape.size() == 2) {
        if ((bias_shape[0] == 1 && bias_shape[1] == matmul_shape[1]) ||
            (bias_shape[0] == matmul_shape[0] && bias_shape[1] == matmul_shape[1])) {
          return true;
        }
      }
    }

    return false;
  }

  /**
   * @brief Fuse MatMul + Add into Gemm
   */
  Status FuseMatMulAdd(Graph& graph, const Node& matmul_node, const Node& add_node,
                      bool& modified, const logging::Logger& logger) const {
    // Get inputs
    auto matmul_inputs = matmul_node.InputDefs();
    if (matmul_inputs.size() != 2) {
      return Status::OK();  // Skip if not standard MatMul
    }

    // Find bias input (the one that's not MatMul output)
    NodeArg* bias_input = nullptr;
    for (auto* input : add_node.InputDefs()) {
      if (input != matmul_node.OutputDefs()[0]) {
        bias_input = input;
        break;
      }
    }

    if (!bias_input) {
      return Status::OK();
    }

    // Create Gemm node
    std::vector<NodeArg*> gemm_inputs = {
        matmul_inputs[0],  // A
        matmul_inputs[1],  // B
        bias_input         // C (bias)
    };

    std::vector<NodeArg*> gemm_outputs = {
        add_node.OutputDefs()[0]  // Y
    };

    // Create Gemm attributes (identity: no transpose, alpha=1, beta=1)
    ONNX_NAMESPACE::AttributeProto trans_a_attr;
    trans_a_attr.set_name("transA");
    trans_a_attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    trans_a_attr.set_i(0);

    ONNX_NAMESPACE::AttributeProto trans_b_attr;
    trans_b_attr.set_name("transB");
    trans_b_attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    trans_b_attr.set_i(0);

    ONNX_NAMESPACE::AttributeProto alpha_attr;
    alpha_attr.set_name("alpha");
    alpha_attr.set_type(ONNX_NAMESPACE::AttributeProto::FLOAT);
    alpha_attr.set_f(1.0f);

    ONNX_NAMESPACE::AttributeProto beta_attr;
    beta_attr.set_name("beta");
    beta_attr.set_type(ONNX_NAMESPACE::AttributeProto::FLOAT);
    beta_attr.set_f(1.0f);

    NodeAttributes gemm_attrs;
    gemm_attrs["transA"] = trans_a_attr;
    gemm_attrs["transB"] = trans_b_attr;
    gemm_attrs["alpha"] = alpha_attr;
    gemm_attrs["beta"] = beta_attr;

    // Add Gemm node
    Node& gemm_node = graph.AddNode(
        graph.GenerateNodeName("TelumFusedGemm"),
        "Gemm",
        "Fused MatMul+Add for Telum EP",
        gemm_inputs,
        gemm_outputs,
        &gemm_attrs,
        kOnnxDomain);

    // Copy metadata
    gemm_node.SetExecutionProviderType(onnxruntime::kTelumExecutionProvider);

    // Remove old nodes
    graph_utils::RemoveNodeOutputEdges(graph, add_node);
    graph.RemoveNode(add_node.Index());
    graph.RemoveNode(matmul_node.Index());

    modified = true;

    LOGS(logger, INFO) << "Telum EP: Fused MatMul + Add into Gemm";

    return Status::OK();
  }
};

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
