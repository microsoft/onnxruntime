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
                         int64_t qkv_second_dim,     // Number of query heads, e.g., 24
                         int64_t qkv_third_dim,  // Size per head, e.g., 64
                         const ONNX_NAMESPACE::TensorProto* q_tensor,
                         const ONNX_NAMESPACE::TensorProto* k_tensor,
                         const ONNX_NAMESPACE::TensorProto* v_tensor,
                         [[maybe_unused]] bool is_matmul) {
  assert(nullptr != q_tensor);
  assert(nullptr != k_tensor);
  assert(nullptr != v_tensor);

  Initializer q_initializer(*q_tensor, graph.ModelPath());
  Initializer k_initializer(*k_tensor, graph.ModelPath());
  Initializer v_initializer(*v_tensor, graph.ModelPath());
  auto data_type = q_tensor->data_type();

  int64_t single_tensor_output_dim = qkv_second_dim * qkv_third_dim;  // For example, 24 * 64 = 1536.
  int64_t total_output_dim = 3 * single_tensor_output_dim;            // Because we have 3 tensors.

  // Create the new merged initializer for packed QKV.
  ONNX_NAMESPACE::TensorProto initializer;
  initializer.set_name(graph.GenerateNodeArgName("qkv_weights"));

  // Set the shape to (input_dim, total_dim).
  initializer.add_dims(input_dim);
  initializer.add_dims(total_output_dim);
  initializer.set_data_type(data_type);

  // ----- Use the data() API to retrieve the tensor data -----
  // Get the number of elements (each element is a uint8_t).
  size_t q_elements = q_initializer.size();
  size_t k_elements = k_initializer.size();
  size_t v_elements = v_initializer.size();

  // Get pointers to the underlying data.
  const uint8_t* q_data = q_initializer.data<uint8_t>();
  const uint8_t* k_data = k_initializer.data<uint8_t>();
  const uint8_t* v_data = v_initializer.data<uint8_t>();

  // Merge the data into one vector.
  std::vector<uint8_t> merged_data;
  size_t element_count = q_elements + k_elements + v_elements;
  [[maybe_unused]] size_t element_count2 = input_dim * total_output_dim;

  merged_data.reserve(element_count);

  optimizer_utils::MergeMatMulWeights<uint8_t>(q_data, k_data, v_data, merged_data, input_dim, single_tensor_output_dim);

  // Convert the merged data to a string and set it as the raw data for the merged tensor.
  utils::SetRawDataInTensorProto(initializer, merged_data.data(), gsl::narrow<size_t>(element_count) * sizeof(uint8_t));


  // ----- Debug printing to verify the merged data -----
  std::cout << "Q tensor pointer (via data API): " << static_cast<const void*>(q_data) << std::endl;
  std::cout << "K tensor pointer (via data API): " << static_cast<const void*>(k_data) << std::endl;

  std::cout << "Merged QKV weight tensor:" << std::endl;
  std::cout << "  Data type: " << data_type << std::endl;
  std::cout << "  Shape: (" << input_dim << ", " << total_output_dim << ")" << std::endl;
  std::cout << graph.ModelPath() << std::endl;

  // Print a snippet of the merged tensor's data.
  std::cout << "Merged tensor raw_data size: " << initializer.raw_data().size() << std::endl;
  std::cout << "Merged tensor raw_data (first 64 bytes in hex): ";
  //for (size_t i = 0; i < initializer.raw_data().size() && i < 64; ++i) {
  //  std::cout << std::hex << std::setw(2) << std::setfill('0')
  //            << static_cast<unsigned>(static_cast<unsigned char>(initializer.raw_data()[i])) << " ";
 // }
 // std::cout << std::dec << std::endl;

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

    const ONNX_NAMESPACE::TensorProto* cos_cache_tensor = nullptr;
    const ONNX_NAMESPACE::TensorProto* sin_cache_tensor = nullptr;
    onnxruntime::NodeArg* pos_ids_arg = nullptr;
    onnxruntime::NodeArg* cos_cache_arg = nullptr; 
    onnxruntime::NodeArg* sin_cache_arg = nullptr;


    for (auto n = node.InputNodesBegin(); n != node.InputNodesEnd(); ++n) {
      auto& rotary_or_v_node = *graph.GetNode(n->Index());  // get mutable next node

      if ((*n).OpType() == "RotaryEmbedding") {
        for (auto inner = rotary_or_v_node.InputNodesBegin(); inner != rotary_or_v_node.InputNodesEnd(); ++inner) {
          std::cout << "rotary input is " << (*inner).Name() << std::endl;
          auto& mat_mul_node = *graph.GetNode(inner->Index());
          std::cout << "rotary input2 is " << mat_mul_node.Name() << std::endl;
          std::cout << "rotary input3 is " << mat_mul_node.OpType() << std::endl;

          std::cout << "Tensor1 name I am trying to load " << mat_mul_node.InputDefs()[1]->Name() << std::endl;
          std::cout << "Tensor2 name I am trying to load " << mat_mul_node.InputDefs()[2]->Name() << std::endl;

          if (k_proj_tensor == nullptr && !graph.GetInitializedTensor(mat_mul_node.InputDefs()[1]->Name(), k_proj_tensor)) {
            std::cout << "Unable to load K tensor weight" << std::endl;
          }

          if (q_proj_tensor == nullptr && !graph.GetInitializedTensor(mat_mul_node.InputDefs()[1]->Name(), q_proj_tensor)) {
            std::cout << "Unable to load Q tensor weight" << std::endl;
          }
        }
        pos_ids_arg = rotary_or_v_node.MutableInputDefs()[1];

        if (cos_cache_tensor == nullptr && !graph.GetInitializedTensor(rotary_or_v_node.InputDefs()[2]->Name(), cos_cache_tensor)) {
          std::cout << "Unable to load cos cache tensor" << std::endl;
        }

        if (cos_cache_arg == nullptr) {
          cos_cache_arg = rotary_or_v_node.MutableInputDefs()[2];
        }

        if (sin_cache_arg == nullptr) {
          sin_cache_arg = rotary_or_v_node.MutableInputDefs()[3];
        }

        if (sin_cache_tensor == nullptr && !graph.GetInitializedTensor(rotary_or_v_node.InputDefs()[3]->Name(), sin_cache_tensor)) {
          std::cout << "Unable to load sin cache tensor" << std::endl;
        }

        std::cout << "r1" << rotary_or_v_node.InputDefs()[0]->Name() << std::endl;
        std::cout << "r2 " << rotary_or_v_node.InputDefs()[1]->Name() << std::endl;
        std::cout << "r3 " << rotary_or_v_node.InputDefs()[2]->Name() << std::endl;
        std::cout << "r4 " << rotary_or_v_node.InputDefs()[3]->Name() << std::endl;


        // cos and sin
      } else if ((*n).OpType() == "MatMulNBits") {
        if (v_proj_tensor == nullptr && !graph.GetInitializedTensor(rotary_or_v_node.InputDefs()[1]->Name(), v_proj_tensor)) {
          std::cout << "Unable to load V tensor weight" << std::endl;
        }
      } 

      std::cout << rotary_or_v_node.Name() << std::endl;
      std::cout << rotary_or_v_node.OpType() << std::endl;
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

    int64_t hidden_size = q_proj_tensor->dims(0);
   
   onnxruntime::NodeArg& fused_qkv = MergeQkvWeights(graph, hidden_size, q_proj_tensor->dims(1), q_proj_tensor->dims(2), q_proj_tensor, k_proj_tensor, v_proj_tensor, true);
  
          if (pos_ids_arg == nullptr) {
     std::cout << "Position IDs argument is not available; cannot fuse GroupQueryAttention." << std::endl;
     continue;
   }

       [[maybe_unused]] const std::array gqa_input_defs{pos_ids_arg, &fused_qkv, cos_cache_arg, sin_cache_arg};
    
    
   // Now add the fused GroupQueryAttention node.
   Node& gqa_node = graph.AddNode(graph.GenerateNodeName("GroupQueryAttention"),
                                  "GroupQueryAttention",
                                  "Fused GroupQueryAttention subgraphs",
                                  gqa_input_defs,  
                                  {},
                                  {},
                                  kMSDomain);

     gqa_node.SetExecutionProviderType(node.GetExecutionProviderType());
     


   [[maybe_unused]] int sodjsapidjad = 1;
       [[maybe_unused]] int sodjsapidjad2 = 1;

              [[maybe_unused]] int sodjsapidjad3 = 1;


  }

  return Status::OK();
}
}  // namespace onnxruntime
