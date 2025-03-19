// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/group_query_attention_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

   NodeArg& MergeQkvWeights(Graph& graph,
                         int64_t input_dim,     // For example, 3072
                         int64_t num_heads,     // Number of query heads, e.g., 24
                         int64_t head_size,     // Size per head, e.g., 64
                         int64_t kv_num_heads,  // Number of key/value heads, e.g., 24
                         const ONNX_NAMESPACE::TensorProto* q_tensor,
                         const ONNX_NAMESPACE::TensorProto* k_tensor,
                         const ONNX_NAMESPACE::TensorProto* v_tensor,
                         [[maybe_unused]] bool is_matmul) {
  // Ensure valid input tensors.
  assert(nullptr != q_tensor);
  assert(nullptr != k_tensor);
  assert(nullptr != v_tensor);

  // Initialize data access for Q, K, and V.
  [[maybe_unused]] Initializer q_initializer(*q_tensor, graph.ModelPath());
  [[maybe_unused]] Initializer k_initializer(*k_tensor, graph.ModelPath());
  [[maybe_unused]] Initializer v_initializer(*v_tensor, graph.ModelPath());
  auto data_type = q_tensor->data_type();

  // Compute the flattened output dimensions:
  // Q projection dimension = num_heads * head_size.
  // K/V projection dimension = kv_num_heads * head_size.
  int64_t q_dim = num_heads * head_size;      // For example, 24 * 64 = 1536.
  int64_t kv_dim = kv_num_heads * head_size;  // For example, 24 * 64 = 1536.
  int64_t total_dim = q_dim + 2 * kv_dim;     // 1536 + 2*1536 = 4608.

  // Create the new merged initializer for packed QKV.
  ONNX_NAMESPACE::TensorProto initializer;
  initializer.set_name(graph.GenerateNodeArgName("qkv_weights"));

  // Set the shape to (input_dim, total_dim).
  initializer.add_dims(input_dim);
  initializer.add_dims(total_dim);
  initializer.set_data_type(data_type);

  // Assume that each tensor provides its data via raw_data.
  // Each of Q, K, and V has input_dim * (num_heads * head_size) elements.
  // We now concatenate Q, then K, then V data.
  std::string merged_raw;
  merged_raw.reserve(q_tensor->raw_data().size() +
                     k_tensor->raw_data().size() +
                     v_tensor->raw_data().size());
  merged_raw.append(q_tensor->raw_data());
  merged_raw.append(k_tensor->raw_data());
  merged_raw.append(v_tensor->raw_data());
  initializer.set_raw_data(merged_raw);

  std::cout << "Merged QKV weight tensor:" << std::endl;
  std::cout << "  Data type: " << data_type << std::endl;
  std::cout << "  Shape: (" << input_dim << ", " << total_dim << ")" << std::endl;
  std::cout << graph.ModelPath() << std::endl;

  return graph_utils::AddInitializer(graph, initializer);
}


Status GroupQueryAttentionFusion::ApplyImpl(
    Graph& graph,
    bool& modified,
    int graph_level,
    const logging::Logger& logger) const {

GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (const auto& ep : GetCompatibleExecutionProviders()) {
    std::cout << std::string(ep) << std::endl;
  }

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& node = *node_ptr;

    std::cout << node.OpType() << std::endl;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "GroupQueryAttention", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
        continue;
    }

    auto& inputs = node.MutableInputDefs();

    //  auto& mul_node = *graph.GetNode(div_node.OutputNodesBegin()->Index());  // get mutable next node

    std::cout << inputs.size() << std::endl;

    for (auto input : inputs) {
      std::cout << input->Name() << std::endl;
      std::cout << *input->Type() << std::endl;
    }

    std::cout << "-----------------" << std::endl;

    const ONNX_NAMESPACE::TensorProto* k_proj_tensor = nullptr;
    const ONNX_NAMESPACE::TensorProto* q_proj_tensor = nullptr;
    const ONNX_NAMESPACE::TensorProto* v_proj_tensor = nullptr;


    for (auto n = node.InputNodesBegin(); n != node.InputNodesEnd(); ++n) {
      auto& mul_node = *graph.GetNode(n->Index());  // get mutable next node

      if ((*n).OpType() == "RotaryEmbedding") {
        for (auto inner = mul_node.InputNodesBegin(); inner != mul_node.InputNodesEnd(); ++inner) {
          std::cout << "rotary input is " << (*inner).Name() << std::endl;
          auto& mat_mul_node = *graph.GetNode(inner->Index());
          std::cout << "rotary input2 is " << mat_mul_node.Name() << std::endl;
          std::cout << "rotary input3 is " << mat_mul_node.OpType() << std::endl;

          std::cout << "Tensor name I am trying to load " << mat_mul_node.InputDefs()[1]->Name() << std::endl;

          if (k_proj_tensor == nullptr && !graph.GetInitializedTensor(mat_mul_node.InputDefs()[1]->Name(), k_proj_tensor)) {
            std::cout << "Unable to load K tensor weight" << std::endl;
          }

          if (q_proj_tensor == nullptr && !graph.GetInitializedTensor(mat_mul_node.InputDefs()[1]->Name(), q_proj_tensor)) {
            std::cout << "Unable to load Q tensor weight" << std::endl;
          }
        }
      } else if ((*n).OpType() == "MatMulNBits") {
        if (v_proj_tensor == nullptr && !graph.GetInitializedTensor(mul_node.InputDefs()[1]->Name(), v_proj_tensor)) {
          std::cout << "Unable to load V tensor weight" << std::endl;
        }
      } 

      std::cout << mul_node.Name() << std::endl;
      std::cout << mul_node.OpType() << std::endl;
    }

    for (auto* input_def : node.InputDefs()) {
      // Try to find a node that produces this input
      const Node* producer = graph.GetProducerNode(input_def->Name());
      if (producer != nullptr) {
        std::cout << "Input \"" << input_def->Name() << "\" is produced by node: " << producer->Name() << std::endl;
      } else {
        std::cout << "Input \"" << input_def->Name() << "\" is not produced by any node (it is likely an initializer or constant)" << std::endl;
      }
    }

    std::cout << "K tensor Dimensions: ";
    for (int i = 0; i < k_proj_tensor->dims_size(); ++i) {
      std::cout << k_proj_tensor->dims(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Q tensor Dimensions: ";
    for (int i = 0; i < q_proj_tensor->dims_size(); ++i) {
      std::cout << q_proj_tensor->dims(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "V tensor Dimensions: ";
    for (int i = 0; i < v_proj_tensor->dims_size(); ++i) {
      std::cout << v_proj_tensor->dims(i) << " ";
    }

    // num of heads * head size
   int64_t hidden_size = q_proj_tensor->dims(1) * q_proj_tensor->dims(2);

   [[maybe_unused]] onnxruntime::NodeArg& abc = MergeQkvWeights(graph, hidden_size, 24, 64, 24, q_proj_tensor, k_proj_tensor, v_proj_tensor, true);
  }

  return Status::OK();
}
}  // namespace onnxruntime
