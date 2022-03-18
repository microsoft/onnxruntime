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

#define PARSER_DEBUG 0

// Open this to output info to debug
#if PARSER_DEBUG
#define PARSER_MSG(_expr) _expr
#else
#define PARSER_MSG(_expr)
#endif

#undef PARSER_DEBUG

using namespace ONNX_NAMESPACE;

// First: graph node index. Second: corresponding pattern node.
typedef std::pair<onnxruntime::NodeIndex, const onnxruntime::Node*> PNN;
typedef std::function<bool(const onnxruntime::Node*, const onnxruntime::Node*, const onnxruntime::Graph&, const onnxruntime::Graph&)> customized_function;

namespace onnxruntime {
namespace training {

class PNode {
 public:
  PNode(std::string op_type, std::vector<std::string> input_args_name,
        std::vector<std::string> output_args_name, std::string name = "",
        std::vector<std::string> domains = {}, std::vector<int> versions = {},
        std::vector<AttributeProto> attributes = {}) : op_type(op_type),
                                                       input_args_name(input_args_name),
                                                       output_args_name(output_args_name),
                                                       node_name(name),
                                                       domains(domains),
                                                       versions(versions),
                                                       attributes(attributes) {}

  std::string GetNodeName() {
    bool need_construct = node_name.empty();
    if (need_construct) {
      node_name += op_type + "_";
    }
    if (need_construct) {
      for (size_t i = 0; i < input_args_name.size(); i++) {
        node_name += input_args_name[i] + "_";
      }
      for (size_t i = 0; i < output_args_name.size(); i++) {
        node_name += output_args_name[i] + "_";
      }
    }
    return node_name;
  }

  std::unordered_set<std::string> GetValidDomains() { return valid_domains; }

  std::unordered_set<int> GetValidVersions() { return valid_versions; }

  NodeDef GetNodeDef() {
    auto IA = [](const std::string& argSuffix, const TypeProto* type_proto = nullptr) {
      return ArgDef(argSuffix, type_proto);
    };
    std::string domain = domains.empty() ? "" : domains[0];
    int version = versions.empty() ? 9 : versions[0];
    std::vector<ArgDef> input_args(input_args_name.size());
    std::vector<ArgDef> output_args(output_args_name.size());
    // give a name if the name is not given by user
    for (size_t i = 0; i < input_args_name.size(); i++) {
      input_args[i] = IA(input_args_name[i]);
    }
    for (size_t i = 0; i < output_args_name.size(); i++) {
      output_args[i] = IA(output_args_name[i]);
    }
    return NodeDef(OpDef(op_type, domain, version), input_args, output_args, attributes, GetNodeName());
  }

 private:
  std::string op_type;
  std::vector<std::string>& input_args_name;
  std::vector<std::string>& output_args_name;
  std::string node_name;
  std::vector<std::string>& domains;
  std::vector<int>& versions;
  std::vector<AttributeProto>& attributes;
  std::unordered_set<std::string> valid_domains;
  std::unordered_set<int> valid_versions;
};

class PatternGraph {
 private:
  std::string name_;  // name of the graph
  std::string root_node;
  bool check_optype, check_domain, check_version;
  std::vector<customized_function> customized_constriants;
  Model model;
  std::vector<NodeDef> nodes;  // node definitions
  std::map<std::string, std::unordered_set<std::string>> domain_map;
  std::map<std::string, std::unordered_set<int>> version_map;

 public:
  PatternGraph(const std::vector<NodeDef>& constants, const std::vector<PNode>& pnodes, const std::string& root_node = "", const std::string& name = "",
               const std::vector<customized_function>& customized_constriants = {})
      : name_(name),
        root_node(root_node),
        check_optype(true),
        check_domain(true),
        check_version(true),
        customized_constriants(customized_constriants),
        model(Model("pattern_model", false, logging::LoggingManager::DefaultLogger())),
        nodes(constants) {
    for (auto pnode : pnodes) {
      nodes.push_back(pnode.GetNodeDef());
      domain_map[pnode.GetNodeName()] = pnode.GetValidDomains();
      version_map[pnode.GetNodeName()] = pnode.GetValidVersions();
    }
  }

  Status TryMatch(Graph& graph, std::vector<PNN>& res,
                  const std::unordered_set<std::string>* p_initializer_names_to_preserve = nullptr);

  // later
  Status TryReplace(Graph& graph, const NodeDef& alternative, std::vector<std::pair<std::string, int>> fusion_inputs,
                    std::vector<std::pair<std::string, int>> fusion_outputs,
                    const std::unordered_set<std::string>* p_initializer_names_to_preserve = nullptr);

  Status to_graph(Graph& graph,
                  const std::unordered_set<std::string>* p_initializer_names_to_preserve = nullptr) {
    GraphAugmenter::GraphDefs graph_defs;
    graph_defs.AddNodeDefs(nodes);
    // graph_defs.AddGraphInputs(input_nodes);
    // graph_defs.AddGraphOutputs(output_nodes);
    return GraphAugmenter::AugmentGraph(graph, graph_defs, p_initializer_names_to_preserve);
  }

  Status to_graph(Model& model,
                  const std::unordered_set<std::string>* p_initializer_names_to_preserve = nullptr) {
    Graph& graph = model.MainGraph();
    if (graph.Nodes().empty()) {
      GraphAugmenter::GraphDefs graph_defs;
      graph_defs.AddNodeDefs(nodes);
      return GraphAugmenter::AugmentGraph(graph, graph_defs, p_initializer_names_to_preserve);
    }
    // graph_defs.AddGraphInputs(input_nodes);
    // graph_defs.AddGraphOutputs(output_nodes);
    return Status::OK();
  }

  std::string name() const { return name_; }

  PatternGraph& add_node(NodeDef node) {
    nodes.push_back(node);
    return *this;
  }

  PatternGraph& add_nodes(const std::vector<NodeDef>& nodes) {
    this->nodes.insert(this->nodes.end(), nodes.begin(), nodes.end());
    return *this;
  }

  PatternGraph& add_customized_constriant(customized_function func) {
    customized_constriants.push_back(func);
    return *this;
  }

  PatternGraph& disable_optype_check() {
    check_optype = false;
    return *this;
  }

  PatternGraph& disable_domain_check() {
    check_domain = false;
    return *this;
  }

  PatternGraph& disable_version_check() {
    check_version = false;
    return *this;
  }

 private:
  bool node_equal_properties(const Node* g, const Node* p, const Graph& target,
                             const Graph& pattern);
  bool find_match(const Node* g, const Node* p, std::unordered_set<const Node*>& graph_path, std::unordered_set<const Node*>& pattern_path,
                  std::vector<PNN>& matched, const Graph& target, const Graph& pattern);
};

namespace GraphParser {

// Add dangling nodedef, which helps to construct the graph but will not be treated as a part of pattern
template <typename T>
NodeDef Constant(const std::string& name, T value = T(0), const std::vector<int64_t>& shape = {1}) {
  auto ScalarTensorProto = [](T value, std::vector<int64_t> shape) {
    // ORT_ENFORCE(shape.size() == 0 || (shape.size() == 1 && shape[0] == 1));
    auto t_proto = ONNX_NAMESPACE::ToTensor<T>(value);
    for (auto dim : shape) {
      t_proto.add_dims(dim);
    }

    return t_proto;
  };

  auto t_proto = ScalarTensorProto(value, shape);
  return NodeDef("Constant",
                 {},
                 {ArgDef(name, nullptr)},
                 {ONNX_NAMESPACE::MakeAttribute("value", t_proto)});
}

inline bool HasSingleSpeciefiedConstantValue(const Graph& graph, const Node* node, double value) {
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

inline int GetConstantInitializerCount(const Graph& graph, const Node* node) {
  int res = 0;
  for (size_t i = 0; i < node->InputDefs().size(); i++) {
    res += graph_utils::IsConstantInitializer(graph, node->InputDefs()[i]->Name());
  }
  return res;
}

inline const Node* GetNodeOfPatternNodeName(const Graph& graph, const std::vector<PNN>& collection, const std::string& name) {
  for (auto [idx, pnode] : collection) {
    if (pnode->Name() == name) {
      return graph.GetNode(idx);
    }
  }
  return nullptr;
}

inline NodeArg* GetNodeArgWithName(Graph& graph, const std::vector<PNN>& collection, std::string name, int idx) {
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