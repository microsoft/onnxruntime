
#include "core/graph/graph_utils.h"

namespace onnxruntime {

namespace utils {
// fusion is only done for ONNX domain ops
bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       const std::string& op_type,
                                       ONNX_NAMESPACE::OperatorSetVersion version,
                                       const std::string& domain) {
  if (node.OpType() != op_type ||
      node.Op()->Deprecated() || node.Op()->SinceVersion() != version ||
      (!node.Domain().empty() && node.Domain() != domain)) {
    return false;
  }
  return true;
}

Status ForAllMutableSubgraphs(Graph& graph, std::function<Status(Graph&)> func) {
  Status status = Status::OK();

  for (auto& node : graph.Nodes()) {
    for (auto& attribute : node.GetAttributes()) {
      auto& name = attribute.first;
      auto& proto = attribute.second;

      // check if it has a subgraph
      if (proto.has_g()) {
        Graph* subgraph = node.GetMutableGraphAttribute(name);
        ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

        status = func(*subgraph);
        ORT_RETURN_IF_ERROR(status);

        // recurse
        status = ForAllMutableSubgraphs(*subgraph, func);
        ORT_RETURN_IF_ERROR(status);
      }
    }
  }

  return status;
}

Status ForAllSubgraphs(const Graph& graph, std::function<Status(const Graph&)> func) {
  Status status = Status::OK();

  for (auto& node : graph.Nodes()) {
    for (auto& attribute : node.GetAttributes()) {
      auto& name = attribute.first;
      auto& proto = attribute.second;

      // check if it has a subgraph
      if (proto.has_g()) {
        const Graph* subgraph = node.GetGraphAttribute(name);
        ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

        status = func(*subgraph);
        ORT_RETURN_IF_ERROR(status);

        // recurse
        status = ForAllSubgraphs(*subgraph, func);
        ORT_RETURN_IF_ERROR(status);
      }
    }
  }

  return status;
}

bool IsSingleInSingleOutNode(const Node& node) {
  return node.GetInputEdgesCount() == 1 && node.GetOutputEdgesCount() == 1;
}

const onnx::AttributeProto* GetNodeAttribute(
    const Node& node, const std::string& attr_name) {
  const auto& attrs = node.GetAttributes();
  const auto iter = attrs.find(attr_name);
  return iter == attrs.end() ? nullptr : &iter->second;
}

bool RemoveNodeFromPath(Graph& graph, Node& node) {
  if (!IsSingleInSingleOutNode(node)) {
    return false;
  }
  // Get input/output edges, nodes, and node args.
  const Node::EdgeEnd& input_edge = *node.InputEdgesBegin();
  const Node::EdgeEnd& output_edge = *node.OutputEdgesBegin();

  // Remove output edge.
  graph.RemoveEdge(node.Index(), output_edge.GetNode().Index(),
                   output_edge.GetSrcArgIndex(), output_edge.GetDstArgIndex());

  // Remove node (this will remove the input edge too).
  graph.RemoveNode(node.Index());

  // Add new edge connecting the input with the output nodes directly.
  graph.AddEdge(input_edge.GetNode().Index(), output_edge.GetNode().Index(),
                input_edge.GetSrcArgIndex(), output_edge.GetDstArgIndex());

  return true;
}

}  // namespace utils
}  // namespace onnxruntime
