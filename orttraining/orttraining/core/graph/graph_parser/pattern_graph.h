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

#define PARSER_DEBUG 1

// Open this to output info to debug
#if PARSER_DEBUG
#define PARSER_MSG(_expr) _expr
#else
#define PARSER_MSG(_expr)
#endif

#undef PARSER_DEBUG

namespace onnxruntime {
namespace training {

// First: graph node index. Second: corresponding pattern node.
typedef std::pair<onnxruntime::NodeIndex, const onnxruntime::Node*> PNN;
typedef std::function<bool(const onnxruntime::Node*, const onnxruntime::Node*, const onnxruntime::Graph&, const onnxruntime::Graph&)> customized_function;

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
  std::string root_node_;
  bool check_optype, check_domain, check_version, check_path;
  std::vector<customized_function> customized_constriants;
  Model model;
  std::vector<NodeDef> nodes;  // node definitions
  std::map<std::string, std::unordered_set<std::string>> domain_map;
  std::map<std::string, std::unordered_set<int>> version_map;

 public:
  PatternGraph(const std::vector<NodeDef>& constants, const std::vector<PNode>& pnodes, const std::string& root_node = "", const std::string& name = "",
               const std::vector<customized_function>& customized_constriants = {})
      : name_(name),
        root_node_(root_node),
        check_optype(true),
        check_domain(true),
        check_version(true),
        check_path(true),
        customized_constriants(customized_constriants),
        nodes(constants) {
    ort_model_ptr_ = std::make_unique<Model>("pattern_model", false, logging::LoggingManager::DefaultLogger());
    for (auto pnode : pnodes) {
      nodes.push_back(pnode.GetNodeDef());
      domain_map[pnode.GetNodeName()] = pnode.GetValidDomains();
      version_map[pnode.GetNodeName()] = pnode.GetValidVersions();
    }

    ORT_ENFORCE(ToGraphInternal().IsOK());
  }

  std::string name() const { return name_; }

  // TODO: should hide this in private field.
  const std::string& root_node() const { return root_node_; }

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

  PatternGraph& disable_path_check() {
    check_path = false;
    return *this;
  }

  Graph& GetGraph() {
    return ort_model_ptr_->MainGraph();
  }

  bool find_match(const Node* g, const Node* p, std::unordered_set<const Node*>& graph_path,
                  std::unordered_set<const Node*>& pattern_path, std::unordered_map<const Node*, const Node*>& path_map,
                  std::vector<PNN>& matched, const Graph& target, const Graph& pattern);

 private:
  bool node_equal_properties(const Node* g, const Node* p, const Graph& target,
                             const Graph& pattern);

  Status ToGraphInternal() {
    // Todo: p_initializer_names_to_preserve should contains all names in the pattern graphs.
    const std::unordered_set<std::string>* p_initializer_names_to_preserve = nullptr;
    GraphAugmenter::GraphDefs graph_defs;
    graph_defs.AddNodeDefs(nodes);
    ORT_ENFORCE(GraphAugmenter::AugmentGraph(ort_model_ptr_->MainGraph(), graph_defs, p_initializer_names_to_preserve).IsOK());

    return Status::OK();
  }

  std::unique_ptr<Model> ort_model_ptr_;
  bool ort_graph_ready_;
};

}  // namespace training
}  // namespace onnxruntime