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

namespace onnxruntime {
namespace training {

class PatternGraph {
 private:
  std::string name_;           // name of the graph
  std::vector<NodeDef> nodes;  // node definitions
  std::string root_node;

  /*
  * Constriants of input or output of nodes. For instance, as shown below, the "other subgraph" need to be indicated.
       Gather (indice=0)    Gather (indice=1)--+
       |                    |               |
    Unsqueeze            Unsqueeze          |
         \             /                    |
          \           /                  [other subgraph]
           \         /                      |
             Concat                         |
               |                            |
               +----------------------------+--+
  */
  std::unordered_map<std::string, int> input_count_constriants;
  std::unordered_map<std::string, int> output_count_constriants;

  // revise
  std::unordered_set<std::string> valid_types;  // valid types of this pattern
  std::map<std::string, std::vector<std::string>> domain_constriants;
  std::map<std::string, std::vector<int>> version_constriants;
  std::vector<std::function<bool(const Graph&)>> customized_constriants;

  // constiants of valid providers, which seems to be supposed to be defined as a constant?
  const InlinedHashSet<std::string_view> compatible_provider_types;

  Model model;

 public:
  PatternGraph(const std::vector<NodeDef>& nodes, const std::string& root_node = "", const std::string& name = "",
               const std::unordered_map<std::string, int>& input_count_constriants = {{}},
               const std::unordered_map<std::string, int>& output_count_constriants = {{}},
               const std::vector<std::string>& valid_types = {},
               const std::map<std::string, std::vector<std::string>>& valid_domains = {{}},
               const std::map<std::string, std::vector<int>>& valid_versions = {{}},
               const std::vector<std::function<bool(const Graph&)>>& customized_constriants = {{}})
      : name_(name),
        nodes(nodes),
        root_node(root_node),
        input_count_constriants(input_count_constriants),
        output_count_constriants(output_count_constriants),
        valid_types(std::unordered_set<std::string>(valid_types.begin(), valid_types.end())),
        domain_constriants(valid_domains),
        version_constriants(valid_versions),
        customized_constriants(customized_constriants),
        model(Model("pattern_model", false, logging::LoggingManager::DefaultLogger())) {
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
    GraphAugmenter::GraphDefs graph_defs;
    graph_defs.AddNodeDefs(nodes);
    // graph_defs.AddGraphInputs(input_nodes);
    // graph_defs.AddGraphOutputs(output_nodes);
    return GraphAugmenter::AugmentGraph(graph, graph_defs, p_initializer_names_to_preserve);
  }

  std::string name() const { return name_; }

  const PatternGraph& add_node(NodeDef node) {
    nodes.push_back(node);
    return *this;
  }

  const PatternGraph& add_nodes(const std::vector<NodeDef>& nodes) {
    this->nodes.insert(this->nodes.end(), nodes.begin(), nodes.end());
    return *this;
  }

  const PatternGraph& add_input_count_constiant(std::string name, int count) {
    input_count_constriants[name] = count;
    return *this;
  }

  const PatternGraph& add_output_count_constiant(std::string name, int count) {
    output_count_constriants[name] = count;
    return *this;
  }

  const PatternGraph& add_type_constriant(std::string name, const std::vector<std::string>& types) {
    if (!valid_types.count(name)) {
      valid_types.insert(types.begin(), types.end());
    }
    return *this;
  }
  const PatternGraph& add_type_constriant(std::string name, std::string type) {
    if (!valid_types.count(name)) {
      valid_types.insert(type);
    }
    return *this;
  }

  const PatternGraph& add_domain_constriant(std::string name, const std::vector<std::string>& domains) {
    if (domain_constriants.count(name)) {
      domain_constriants[name].insert(domain_constriants[name].end(), domains.begin(), domains.end());
    } else {
      domain_constriants[name] = std::vector<std::string>(domains);
    }
    return *this;
  }
  const PatternGraph& add_domain_constriant(std::string name, std::string domain) {
    if (domain_constriants.count(name)) {
      domain_constriants[name].push_back(domain);
    } else {
      domain_constriants[name] = std::vector<std::string>(1, domain);
    }
    return *this;
  }

  const PatternGraph& add_version_constriant(std::string name, const std::vector<int>& versions) {
    if (version_constriants.count(name)) {
      version_constriants[name].insert(version_constriants[name].end(), versions.begin(), versions.end());
    } else {
      version_constriants[name] = std::vector<int>(versions);
    }
    return *this;
  }
  const PatternGraph& add_version_constriant(std::string name, int version) {
    if (version_constriants.count(name)) {
      version_constriants[name].push_back(version);
    } else {
      version_constriants[name] = std::vector<int>(1, version);
    }
    return *this;
  }

  const PatternGraph& add_customized_constriant(std::function<bool(const Graph&)> func) {
    customized_constriants.push_back(func);
    return *this;
  }
};

namespace GraphParser {

// Add dangling nodedef, which helps to construct the graph but will not be treated as a part of pattern
inline NodeDef GetDanglingNode(const std::string& name) {
  auto ScalarTensorProto = [](double value, std::vector<int64_t> shape) {
    ORT_ENFORCE(shape.size() == 0 || (shape.size() == 1 && shape[0] == 1));
    auto t_proto = ONNX_NAMESPACE::ToTensor<double>(value);
    for (auto dim : shape) {
      t_proto.add_dims(dim);
    }

    return t_proto;
  };

  auto t_proto = ScalarTensorProto(.0, {1});
  return NodeDef("Constant",
                 {},
                 {ArgDef(name, nullptr)},
                 {ONNX_NAMESPACE::MakeAttribute("value", t_proto)});
}

// Get a nodedef of pattern
inline NodeDef GetNode(const std::string& op_type, const std::vector<std::string>& input_args_name,
                       const std::vector<std::string>& output_args_name, const NodeAttributes& attributes = NodeAttributes(),
                       const std::string& name = "", int priority = 0) {
  auto IA = [](const std::string& argSuffix, const TypeProto* type_proto = nullptr) {
    return ArgDef(argSuffix, type_proto);
  };

  std::vector<ArgDef> input_args(input_args_name.size());
  std::vector<ArgDef> output_args(output_args_name.size());
  std::string node_name = name;
  // give a name if the name is not given by user
  if (name.empty()) {
    node_name += op_type + "_";
  }
  for (size_t i = 0; i < input_args_name.size(); i++) {
    input_args[i] = IA(input_args_name[i]);
    if (name.empty()) {
      node_name += input_args_name[i] + "_";
    }
  }
  for (size_t i = 0; i < output_args_name.size(); i++) {
    output_args[i] = IA(output_args_name[i]);
    if (name.empty()) {
      node_name += output_args_name[i] + "_";
    }
  }
  return NodeDef(op_type, input_args, output_args, attributes, node_name, priority);
}

}  // namespace GraphParser

}  // namespace training
}  // namespace onnxruntime