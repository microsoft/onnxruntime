#include <cmath>
#include <numeric>
#include <list>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <functional>

#include "orttraining/core/graph/gradient_builder.h"

#include <cmath>
#include <numeric>
#include <list>
#include <stack>

#include "onnx/defs/attr_proto_util.h"
#include "onnx/defs/tensor_proto_util.h"

#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/common/safeint.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/core/graph/gradient_builder_registry.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "core/framework/iexecutor.h"
#include "core/graph/graph_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "orttraining/core/framework/ortmodule_graph_builder.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "core/optimizer/initializer.h"
#include "orttraining/core/optimizer/graph_transformer_utils.h"
#include "orttraining/core/graph/graph_parser/pattern_graph.h"
#include "orttraining/core/graph/graph_parser/matcher.h"

#define PARSER_DEBUG 1

// Open this to output info to debug
#if PARSER_DEBUG
#define PARSER_MSG(_expr) _expr
#else
#define PARSER_MSG(_expr)
#endif

#undef PARSER_DEBUG

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

namespace GraphParser {

Status TryMatch(Graph& target_graph, PatternGraph& _graph, std::vector<PNN>& res) {
  Graph& pattern_graph = _graph.GetGraph();
  GraphViewer graph_viewer(target_graph);
  GraphViewer pattern_viewer(pattern_graph);
  const auto& graph_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  const auto& pattern_topology_list = pattern_viewer.GetNodesInTopologicalOrder();

  Node* pattern_root = nullptr;  // Not real root, only a root specified by user
  if (pattern_topology_list.size() && _graph.root_node().empty()) {
    pattern_root = pattern_graph.GetNode(0);
  }
  for (auto node_index : pattern_topology_list) {
    auto* node = pattern_graph.GetNode(node_index);
    if (strcmp(node->Name().c_str(), _graph.root_node().c_str()) == 0) {
      pattern_root = node;
      break;
    }
  }
  if (!pattern_root) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Pattern root was not found.");
  }

  for (auto node_index : graph_topology_list) {
    auto* node = target_graph.GetNode(node_index);
    res.clear();
    std::unordered_set<const Node*> graph_path, pattern_path;
    std::unordered_map<const Node*, const Node*> path_map;
    if (_graph.find_match(node, pattern_root, graph_path, pattern_path, path_map, res, target_graph, pattern_graph)) {
      res.push_back({node_index, pattern_root});
      return Status::OK();
    }
  }

  return Status(common::ONNXRUNTIME, common::FAIL, "No match for the target graph.");
}

Status TryReplace(Graph& graph, PatternGraph& pattern, const NodeDef& alternative,
                  std::vector<std::pair<std::string, int>> fusion_inputs,
                  std::vector<std::pair<std::string, int>> fusion_outputs) {
  std::vector<PNN> match_results;
  ORT_RETURN_IF_ERROR(TryMatch(graph, pattern, match_results));
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