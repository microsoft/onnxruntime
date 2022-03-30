
#include "orttraining/core/graph/graph_augmenter.h"
#include "orttraining/core/graph/graph_parser/matcher.h"
#include "core/optimizer/initializer.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

namespace GraphParser {

Status TryReplace(Graph& graph, PatternGraph& pattern, const NodeDef& alternative,
                  std::vector<std::pair<std::string, int>> fusion_inputs,
                  std::vector<std::pair<std::string, int>> fusion_outputs) {
  std::vector<PNN> match_results;
  ORT_RETURN_IF_ERROR(pattern.TryMatch(graph, match_results));
  InlinedVector<std::reference_wrapper<Node>> matched_nodes;
  for (auto iter = match_results.rbegin(); iter != match_results.rend(); iter++) {
    auto node = graph.GetNode(iter->first);
    matched_nodes.push_back(*node);
  }

  auto add_node_args_with_name = [&match_results](Graph& graph, std::string name, int idx, std::vector<NodeArg*>& args) {
    GraphViewer viewer(graph);
    // const auto& node_topology_list = viewer.GetNodesInTopologicalOrder();
    for (auto [node_index, pattern_node] : match_results) {
      if (pattern_node->Name() == name) {
        auto target_node = graph.GetNode(node_index);
        args.push_back(target_node->MutableInputDefs()[idx]);
        return Status::OK();
      }
    }
    return Status(common::ONNXRUNTIME, common::FAIL);
  };

  std::vector<NodeArg*> input_args, output_args;
  for (auto item : fusion_inputs) {
    ORT_RETURN_IF_ERROR(add_node_args_with_name(graph, item.first, item.second, input_args));
  }
  for (auto item : fusion_outputs) {
    ORT_RETURN_IF_ERROR(add_node_args_with_name(graph, item.first, item.second, output_args));
  }
  Node& replace_node = graph.AddNode(alternative.name, alternative.op_type, "",
                                     input_args, output_args, {}, alternative.domain.c_str());
  for (auto& attr : alternative.attributes) {
    replace_node.AddAttribute(attr.first, attr.second);
  }
  graph_utils::FinalizeNodeFusion(graph, matched_nodes, replace_node);
  return Status::OK();
}

bool HasSingleSpeciefiedConstantValue(const Graph& graph, const Node* node, double value) {
  if (node->InputDefs().size() != 2) return false;
  bool input0_is_initializer = graph_utils::IsConstantInitializer(graph, node->InputDefs()[0]->Name());
  bool input1_is_initializer = graph_utils::IsConstantInitializer(graph, node->InputDefs()[1]->Name());
  // reject if both or neither inputs are initializers for now
  if (input0_is_initializer == input1_is_initializer) {
    return false;
  }

  const auto* initializer = graph_utils::GetConstantInitializer(graph, node->InputDefs()[input0_is_initializer ? 0 : 1]->Name());

  // if initializer_rank is bigger, the output is expected to be initializer_rank per broadcasting rule,
  // but it won't happen if the case is accepted, thus reject it
  auto initializer_rank = initializer->dims().size();
  const auto* other_input_shape = node->InputDefs()[input0_is_initializer ? 1 : 0]->Shape();
  if (other_input_shape == nullptr || initializer_rank > other_input_shape->dim_size()) {
    return false;
  }

  int32_t data_type = initializer->data_type();
  Initializer add_init(*initializer, graph.ModelPath());
  if (add_init.size() > 1) {
    return false;
  }
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      if (*add_init.data<float>() != value) {
        return false;
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      if (math::halfToFloat(add_init.data<MLFloat16>()->val) != value) {
        return false;
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      if (*add_init.data<double>() != static_cast<double>(value)) {
        return false;
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      if (*add_init.data<int32_t>() != static_cast<int32_t>(value)) {
        return false;
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      if (*add_init.data<int64_t>() != static_cast<int64_t>(value)) {
        return false;
      }
      break;
    default:
      return false;
  }
  return true;
}

int GetConstantInitializerCount(const Graph& graph, const Node* node) {
  int res = 0;
  for (size_t i = 0; i < node->InputDefs().size(); i++) {
    res += graph_utils::IsConstantInitializer(graph, node->InputDefs()[i]->Name());
  }
  return res;
}

Node* GetNodeOfPatternNodeName(const Graph& graph, const std::vector<PNN>& collection, const std::string& name) {
  for (auto [idx, pnode] : collection) {
    if (pnode->Name() == name) {
      return const_cast<Node*>(graph.GetNode(idx));
    }
  }
  return nullptr;
}

NodeArg* GetNodeArgWithName(Graph& graph, const std::vector<PNN>& collection, std::string name, int idx) {
  // const auto& node_topology_list = viewer.GetNodesInTopologicalOrder();
  for (auto [node_index, pattern_node] : collection) {
    if (pattern_node->Name() == name) {
      auto target_node = graph.GetNode(node_index);
      return target_node->MutableInputDefs()[idx];
    }
  }
  return nullptr;
};

template <typename T>
std::vector<T> GetConstantInitializers(const Graph& graph, const Node* node) {
  std::vector<T> res;
  for (auto inp : node->InputDefs()) {
    bool input0_is_initializer = graph_utils::IsInitializer(graph, inp->Name(), false);
    // reject if both or neither inputs are initializers for now
    if (!input0_is_initializer) {
      continue;
    }

    const auto* initializer = graph_utils::GetConstantInitializer(graph, inp->Name());
    Initializer add_init(*initializer, graph.ModelPath());
    res.push_back(*add_init.data<T>());
  }
  return res;
}

}  // namespace GraphParser

}  // namespace training
}  // namespace onnxruntime