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

#define PARSER_LOG \
  LOGS_DEFAULT(INFO)

namespace onnxruntime {
namespace training {

// First: graph node index. Second: corresponding pattern node.
typedef std::pair<onnxruntime::NodeIndex, const onnxruntime::Node*> PNN;

// TODO: we should sperate initializer and constant nodes.
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

struct PNode {
  PNode(std::string op_type,
        std::vector<std::string> input_args_name,
        std::vector<std::string> output_args_name,
        std::string node_name = "",
        std::vector<std::string> domains = {},
        std::vector<int> versions = {},
        std::vector<AttributeProto> attributes = {})
      : op_type_(op_type),
        input_args_name(input_args_name),
        output_args_name(output_args_name),
        domains(domains),
        versions(versions),
        attributes(attributes) {
    if (node_name.empty()) {
      std::stringstream ss;
      ss << op_type_ << "_";

      for (size_t i = 0; i < input_args_name.size(); i++) {
        ss << input_args_name[i] << "_";
      }
      for (size_t i = 0; i < output_args_name.size(); i++) {
        ss << output_args_name[i] << "_";
      }

      node_name_ = ss.str();
    }
  }

  std::string GetNodeName() const {
    return node_name_;
  }

  bool OpTypeEquals(const std::string& op_type) const {
    return op_type_.compare(op_type) == 0;
  }

  bool NameEquals(const std::string& name) const {
    return node_name_.compare(name) == 0;
  }

  bool DomainsContain(const std::string& domain) const {
    return std::find(domains.begin(), domains.end(), domain) != domains.end();
  }

  bool VersionsContain(int version) const {
    return std::find(versions.begin(), versions.end(), version) != versions.end();
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
    return NodeDef(OpDef(op_type_, domain, version), input_args, output_args, attributes, GetNodeName());
  }

 private:
  std::string op_type_;
  std::vector<std::string>& input_args_name;
  std::vector<std::string>& output_args_name;
  std::string node_name_;
  std::vector<std::string>& domains;
  std::vector<int>& versions;
  std::vector<AttributeProto>& attributes;
  std::unordered_set<std::string> valid_domains;
  std::unordered_set<int> valid_versions;
};

struct PatternGraph;

class NodeCompareFunc {
 public:
  virtual bool operator()(const Node* g, const PNode* p, const Graph& /*target*/, const PatternGraph& /*pattern*/) const = 0;
};

class DefaultNodeCompareFunc : public NodeCompareFunc {
 public:
  DefaultNodeCompareFunc(bool skip_optype, bool skip_domain,
                         bool skip_version, bool skip_path)
      : skip_optype_(skip_optype),
        skip_domain_(skip_domain),
        skip_version_(skip_version),
        skip_path_(skip_path) {}

  bool operator()(const Node* g, const PNode* p, const Graph& /*target*/, const PatternGraph& /*pattern*/) const override {
    if (!g && !p)
      return true;
    if (!g && p || !p && g)
      return false;
    if (!skip_optype_ && p->NameEquals(g->OpType())) {
      // PARSER_LOG << "OpType mismatch, "
      //            << "g is: " << g->OpType() << ", p is: " << p->OpType();
      return false;
    }
    if (!skip_domain_ && p->DomainsContain(g->Domain())) {
      // PARSER_LOG << "Domain mismatch, "
      //            << "g is: " << g->Domain() << ", p is: " << p->Domain();
      return false;
    }
    if (!skip_version_ && p->VersionsContain(g->SinceVersion())) {
      // PARSER_LOG << "Version mismatch, "
      //            << "g is: " << g->SinceVersion() << ", p is: " << p->SinceVersion();
      return false;
    }
    return true;
  }

  bool skip_optype_, skip_domain_, skip_version_, skip_path_;
};

// typedef std::function<bool(const Node*, const PNode*, const Graph&, const PatternGraph&)> NodeCompareFunc;

struct PatternGraph {
  PatternGraph(const std::vector<NodeDef>& constants, const std::vector<PNode>& pnodes,
               const std::string& root_node = "", const std::string& name = "")
      : name_(name),
        root_node_(root_node),
        nodes(constants) {
    ort_model_ptr_ = std::make_unique<Model>("pattern_model", false, logging::LoggingManager::DefaultLogger());
    for (auto pnode : pnodes) {
      name_pnode_mapping_[pnode.GetNodeName()] = &pnode;
      nodes.push_back(pnode.GetNodeDef());
      domain_map[pnode.GetNodeName()] = pnode.GetValidDomains();
      version_map[pnode.GetNodeName()] = pnode.GetValidVersions();
    }

    ORT_ENFORCE(ToGraphInternal().IsOK());

    default_compare_func_ = std::make_unique<DefaultNodeCompareFunc>(false, false, false, false);
  }

  const std::string& name() const {
    return name_;
  }

  // TODO: should hide this in private field.
  const std::string& root_node() const { return root_node_; }

  PatternGraph& SetCustomConstraint(std::unique_ptr<NodeCompareFunc> func, std::string node_name = "") {
    if (node_name.empty()) {
      default_compare_func_ = std::move(func);
      custom_constraints_.clear();
    } else {
      custom_constraints_[node_name] = std::move(func);
    }
    return *this;
  }

  // TODO: evaluate should we have them?
  /*
  PatternGraph& add_node(NodeDef node) {
    nodes.push_back(node);
    return *this;
  }

  PatternGraph& add_nodes(const std::vector<NodeDef>& nodes) {
    this->nodes.insert(this->nodes.end(), nodes.begin(), nodes.end());
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
  */

  Graph& GetGraph() {
    return ort_model_ptr_->MainGraph();
  }

  Status TryMatch(Graph& target_graph, std::vector<PNN>& res);

 private:
  Status ToGraphInternal() {
    // Todo: p_initializer_names_to_preserve should contains all names in the pattern graphs.
    const std::unordered_set<std::string>* p_initializer_names_to_preserve = nullptr;
    GraphAugmenter::GraphDefs graph_defs;
    graph_defs.AddNodeDefs(nodes);
    ORT_ENFORCE(GraphAugmenter::AugmentGraph(ort_model_ptr_->MainGraph(), graph_defs, p_initializer_names_to_preserve).IsOK());

    return Status::OK();
  }

  bool FindMatchRecursively(const Node* g, const Node* p,
                          std::unordered_set<const Node*>& graph_path, std::unordered_set<const Node*>& pattern_path,
                          std::unordered_map<const Node*, const Node*>& path_map,
                          std::vector<PNN>& matched, const Graph& target);

  std::unique_ptr<Model> ort_model_ptr_;

  std::string name_;  // name of the graph
  std::string root_node_;

  std::vector<NodeDef> nodes;  // node definitions
  std::map<std::string, std::unordered_set<std::string>> domain_map;
  std::map<std::string, std::unordered_set<int>> version_map;

  std::unordered_map<std::string, const PNode*> name_pnode_mapping_;

  std::unique_ptr<NodeCompareFunc> default_compare_func_;
  std::unordered_map<std::string, std::unique_ptr<NodeCompareFunc>> custom_constraints_;
};

}  // namespace training
}  // namespace onnxruntime