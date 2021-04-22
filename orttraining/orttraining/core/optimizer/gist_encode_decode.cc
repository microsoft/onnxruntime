// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/op.h"
#include "core/optimizer/rewrite_rule.h"
#include "orttraining/core/optimizer/gist_encode_decode.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {
struct GraphEdgeHelper {
  NodeIndex src_node;
  NodeIndex dst_node;
  int src_arg_index;
  int dst_arg_index;
  std::string arg_name;

  GraphEdgeHelper(NodeIndex src_node, NodeIndex dst_node,
                  int src_arg_index, int dst_arg_index, const std::string& arg_name) : src_node(src_node),
                                                                                       dst_node(dst_node),
                                                                                       src_arg_index(src_arg_index),
                                                                                       dst_arg_index(dst_arg_index),
                                                                                       arg_name(arg_name) {}

  static GraphEdgeHelper CreateGraphEdge(const Node& node, const Node::EdgeEnd& edge_end, bool is_input_edge) {
    return is_input_edge
               ? GraphEdgeHelper(edge_end.GetNode().Index(),
                                 node.Index(),
                                 edge_end.GetSrcArgIndex(),
                                 edge_end.GetDstArgIndex(),
                                 node.InputDefs()[edge_end.GetDstArgIndex()]->Name())
               : GraphEdgeHelper(node.Index(),
                                 edge_end.GetNode().Index(),
                                 edge_end.GetSrcArgIndex(),
                                 edge_end.GetDstArgIndex(),
                                 node.InputDefs()[edge_end.GetSrcArgIndex()]->Name());
  }
};

static std::vector<GraphEdgeHelper> GetNodeOutputEdges(const Node& node) {
  std::vector<GraphEdgeHelper> output_edges;
  for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
    output_edges.push_back(GraphEdgeHelper::CreateGraphEdge(node, *it, false));
  }

  return output_edges;
}

bool GistEncodeDecode::AddEncodeDecode(Graph& graph, Node& curr_node, std::string compression_type) const {
  if (curr_node.OutputDefs().size() < 1) {  // min 1 required for gist applicability (one edge connecting a fw node to a bw node)
    return false;
  }

  // Collect output tensors for compression + destination nodes + destination nodes' input edge
  std::vector<GraphEdgeHelper> output_edges = GetNodeOutputEdges(curr_node);
  vector_t lookup_vec = PATTERN_MAP.at(curr_node.OpType());

  typedef int src_arg_idx;
  typedef int dst_arg_idx;
  typedef std::pair<Node*, dst_arg_idx> decode_pair;
  std::unordered_map<src_arg_idx, std::vector<decode_pair>> decode_map;
  for (auto& output_edge : output_edges) {
    Node* node_dst = graph.GetNode(output_edge.dst_node);
    for (auto& lookup_OpType : lookup_vec) {
      if (node_dst->Description() == "Backward pass" && node_dst->OpType() == lookup_OpType) {
        decode_map[output_edge.src_arg_index].push_back(decode_pair(node_dst, output_edge.dst_arg_index));
      }
    }
  }

  if (decode_map.empty()) {
    return false;
  }

  std::string user_compression_type = compression_type;

  // Each element in map corresponds to a stash activation
  for (auto& st_act : decode_map) {
    // Create compressed tensor
    NodeArg* curr_node_output_arg = curr_node.MutableOutputDefs()[st_act.first];
    ONNX_NAMESPACE::TypeProto compressed_tensor;
    compression_type = user_compression_type;

    // Override compression_type for lossless compression case(s) (eg. bool -> Pack1)
    ONNX_NAMESPACE::DataType type_string = curr_node_output_arg->Type();
    if (*type_string == "bool" || *type_string == "tensor(bool)") {
      std::cout << "(Lossless) override compression type to Pack1 for tensor: " << curr_node_output_arg->Name() << std::endl;
      compression_type = "GistPack1";
    }

    if (compression_type == "GistPack1" || compression_type == "GistPack8" || compression_type == "GistPackMsfp15") {
      compressed_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
    } else if (compression_type == "GistPack16") {
      compressed_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    } else {
      assert(0);  // "Gist compression type not supported"
    }

    bool tensor_size_compressed = false;
    for (int i = 0; i < curr_node_output_arg->Shape()->dim_size(); i++) {
      if (curr_node_output_arg->Shape()->dim(i).dim_value() != 0) {
        if (compression_type == "GistPack1" && !tensor_size_compressed) {
          compressed_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value((curr_node_output_arg->Shape()->dim(i).dim_value() + GIST_PACK1_FACTOR - 1) / GIST_PACK1_FACTOR);
          tensor_size_compressed = true;
        } else {
          compressed_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(curr_node_output_arg->Shape()->dim(i).dim_value());
        }
      } else {
        compressed_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(curr_node_output_arg->Shape()->dim(i).dim_param());
      }
    }

    // Create encode/decode nodes
    std::string gist_pair_name = graph.GenerateNodeName(GIST_PAIR_NODE_NAME_BASE);

    std::string encode_node_name = "encode_" + gist_pair_name;
    auto& encode_output_def_compressed_arg = graph.GetOrCreateNodeArg("compr_" + curr_node_output_arg->Name(), &compressed_tensor);
    auto& encode = graph.AddNode(encode_node_name, compression_type + "Encoder", "Encode", {curr_node_output_arg}, {&encode_output_def_compressed_arg}, nullptr, kMSDomain);
    // Nested Gist encoders: Encoders have high priority, and are executed eagerly. Hence, all encoders are assigned the same priority value.
    encode.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_HIGH));

    std::string decode_node_name = "decode_" + gist_pair_name;
    auto& decode_output_def_uncompressed_arg = graph.GetOrCreateNodeArg("uncompr_" + curr_node_output_arg->Name(), curr_node_output_arg->TypeAsProto());
    // Nested Gist decoders: Decoders have low priority, and need to be differentiated. Hence, each decoder is assigned a unqiue priority value.
    int curr_dec_priority = GenerateDecodePriority();
    assert(curr_dec_priority > 0);
    auto& decode = graph.AddNode(decode_node_name, compression_type + "Decoder", "Decode", {&encode_output_def_compressed_arg}, {&decode_output_def_uncompressed_arg}, nullptr, kMSDomain);
    decode.SetPriority(curr_dec_priority);

    // Connect decode node to destination nodes/edges
    for (auto& dest_pair : st_act.second) {
      graph.AddEdge(decode.Index(), dest_pair.first->Index(), 0, dest_pair.second);
    }
  }

  return true;
}

std::vector<std::string> GistEncodeDecode::TargetOpTypes() const noexcept {
  switch (operator_type) {
    case 1:
      return {"Softmax"};
      break;
    case 2:
      return {"Transpose"};
      break;
    case 3:
      return {"Reshape"};
      break;
    case 4:
      return {"Add"};
      break;
    case 5:
      return {"Dropout"};
      break;
    case 6:
      return {"LayerNormalization"};
      break;
    case 7:
      return {"MatMul"};
      break;
    case 8:
      return {"Relu"};
      break;
    case 9:
      return {"Softmax", "Transpose", "Reshape", "Add", "Dropout", "LayerNormalization", "MatMul", "Relu"};
      break;
    default:
      std::cout << "Gist op type not supported" << std::endl;
      return {};
      break;
  }
}

Status GistEncodeDecode::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const {
  if (node.Description() != "Backward pass") {
    if (GistEncodeDecode::AddEncodeDecode(graph, node, compression_type)) {
      LOGS(logger, INFO) << "Gist applied to node name -  " << node.Name() << ", node type - "
                         << node.OpType() << ", of compr type - " << compression_type;
      rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
    }
  }

  return Status::OK();
}

bool GistEncodeDecode::SatisfyCondition(const Graph&, const Node& node, const logging::Logger&) const {
  return node.OutputDefs().size() >= 1;
}

}  // namespace onnxruntime
