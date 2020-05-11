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
static std::vector<GraphEdgeHelper> GetNodeInputEdges(const Node& node) {
  std::vector<GraphEdgeHelper> input_edges;
  for (auto it = node.InputEdgesBegin(), end = node.InputEdgesEnd(); it != end; ++it) {
    input_edges.push_back(GraphEdgeHelper::CreateGraphEdge(node, *it, true));
  }
  return input_edges;
}
bool GistEncodeDecode::AddEncodeDecode(Graph& graph, Node& curr_node, std::string compression_type) const {
  ONNX_NAMESPACE::TypeProto bool_tensor;
  bool_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  if (curr_node.GetOutputEdgesCount() < 1) {
    return false;
  }

  std::vector<GraphEdgeHelper> output_edges = GetNodeOutputEdges(curr_node);

  auto curr_node_output_defs = curr_node.OutputDefs();
  auto curr_node_output_def_name = curr_node_output_defs[0]->Name();
  auto* curr_node_output_arg = graph.GetNodeArg(curr_node_output_def_name);

  std::string encode_node_name = graph.GenerateNodeName(GIST_ENCODER_NODE_NAME_BASE);
  for (int i = 0; i < curr_node_output_arg->Shape()->dim_size(); i++) {
    bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(curr_node_output_arg->Shape()->dim(i).dim_value());
  }
  auto& encode_output_def_compressed_arg = graph.GetOrCreateNodeArg(encode_node_name, &bool_tensor);
  auto& encode_output_def_uncompressed_arg = graph.GetOrCreateNodeArg(encode_node_name + "_identity", curr_node_output_arg->TypeAsProto());
  auto& encode = graph.AddNode(encode_node_name, compression_type + "Encoder", "Encode", {curr_node_output_arg}, {&encode_output_def_uncompressed_arg, &encode_output_def_compressed_arg}, {}, kMSDomain);

  std::string decode_arg_name = graph.GenerateNodeName(GIST_DECODER_NODE_NAME_BASE);
  auto& decode_output_def_uncompressed_arg = graph.GetOrCreateNodeArg(decode_arg_name, curr_node_output_arg->TypeAsProto());
  auto& decode_output_def_dummy_arg = graph.GetOrCreateNodeArg(decode_arg_name + "_late_dec", curr_node_output_arg->TypeAsProto());
  auto& decode = graph.AddNode(decode_arg_name, compression_type + "Decoder", "Decode", {&decode_output_def_dummy_arg, &encode_output_def_compressed_arg}, {&decode_output_def_uncompressed_arg}, {}, kMSDomain);

  bool early_encoding = false;
  bool late_decoding = false;
  for (auto& output_edge : output_edges) {
    Node* node_dst = graph.GetNode(output_edge.dst_node);
    if (node_dst->Description() == "Backward pass" && (node_dst->OpType() == "ReluGrad")) {
      graph.AddEdge(output_edge.src_node, encode.Index(), output_edge.src_arg_index, 0);
      graph.AddEdge(encode.Index(), decode.Index(), 1, 1);
      graph.AddEdge(decode.Index(), output_edge.dst_node, 0, output_edge.dst_arg_index);
      std::vector<GraphEdgeHelper> input_edges_dst = GetNodeInputEdges(*node_dst);
      size_t i = 0;
      while (!late_decoding && i < input_edges_dst.size()) {
        if (graph.GetNode(input_edges_dst[i].src_node)->OpType() != curr_node.OpType()) {
          graph.AddEdge(input_edges_dst[i].src_node, decode.Index(), input_edges_dst[i].src_arg_index, 0);
          late_decoding = true;
        }
        i++;
      }
    } else if (!early_encoding) {
      graph.AddEdge(encode.Index(), output_edge.dst_node, 0, output_edge.dst_arg_index);
      early_encoding = true;
    }
  }
  return true;
}
Status GistEncodeDecode::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  if (GistEncodeDecode::AddEncodeDecode(graph, node, "GistBinarize")) {
    rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  }

  return Status::OK();
}

bool GistEncodeDecode::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  return graph_utils::CanRemoveNode(graph, node, logger);
}

}  // namespace onnxruntime
