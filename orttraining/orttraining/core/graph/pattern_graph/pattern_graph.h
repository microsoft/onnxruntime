// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/contrib_ops/onnx_function_util.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"

#define PARSER_LOG \
  LOGS_DEFAULT(INFO)

namespace onnxruntime {
namespace training {

// First: graph node index. Second: corresponding pattern node.
typedef std::pair<onnxruntime::NodeIndex, const onnxruntime::Node*> PatternMatchPair;

struct PatternMatchResult {
 public:
  explicit PatternMatchResult(Graph& target_graph) : target_graph_(target_graph) {}

  Node* GetTargetNodeWithName(std::string name) const {
    auto res = mapping_.find(name);
    ORT_ENFORCE(res != mapping_.end(), "No target node has cooresponding name %s in pattern graph", name);
    return target_graph_.GetNode(res->second.first);
  }

  NodeIndex GetTargetNodeIdxWithName(std::string name) const {
    auto res = mapping_.find(name);
    ORT_ENFORCE(res != mapping_.end(), "No target node has cooresponding name %s in pattern graph", name);
    return res->second.first;
  }

  PatternMatchPair GetMatchPairWithName(std::string name) const {
    auto res = mapping_.find(name);
    ORT_ENFORCE(res != mapping_.end(), "No node with name %s in pattern graph", name);
    return res->second;
  }

  const onnxruntime::Node* GetPatternNodeWithName(std::string name) const {
    auto res = mapping_.find(name);
    ORT_ENFORCE(res != mapping_.end(), "No node with name %s in pattern graph", name);
    return res->second.second;
  }

  std::map<std::string, PatternMatchPair> GetMatchMap() const {
    return mapping_;
  }

  Graph& GetTargetGraph() const {
    return target_graph_;
  }

  void InsertMatchMappings(std::map<std::string, PatternMatchPair> results) {
    mapping_.insert(results.begin(), results.end());
  }

  void AddMatchMapping(std::string pattern_node_name, PatternMatchPair result) {
    mapping_[pattern_node_name] = result;
  }

  void Clear() {
    mapping_.clear();
  }

 private:
  Graph& target_graph_;
  std::map<std::string, PatternMatchPair> mapping_;
};

class PatternType {
 public:
  enum class PatternTypeCategory {
    Integer,
    Float,
    UnsignedInteger
  };

 public:
  PatternType(PatternTypeCategory category) {
    switch (category) {
      case PatternTypeCategory::Integer:
        Init({ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
              ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
              ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16,
              ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8});
        break;
      case PatternTypeCategory::Float:
        Init({ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
              ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE,
              ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16,
              ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16});
        break;
      case PatternTypeCategory::UnsignedInteger:
        Init({ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64,
              ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32,
              ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16,
              ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8});
        break;
      default:
        break;
    }
  }

  PatternType(const std::vector<ONNX_NAMESPACE::TensorProto_DataType>& types) : types_(types) {}

  ONNX_NAMESPACE::TensorProto_DataType GetDefaultType() const {
    ORT_ENFORCE(size(), "Empty type list in PatternType.");
    return types_.at(0);
  }

  std::vector<ONNX_NAMESPACE::TensorProto_DataType> GetTypes() const {
    return types_;
  }

  size_t size() const {
    return types_.size();
  }

 private:
  void Init(const std::vector<ONNX_NAMESPACE::TensorProto_DataType>& types) {
    types_ = types;
  }

 private:
  std::vector<ONNX_NAMESPACE::TensorProto_DataType> types_;
};

/* 
* \brief Dangling node
*/
class IArg {
 public:
  IArg(const std::string& name, const PatternType& type, std::vector<int> shape = {1},
       bool is_dangling = true, bool is_constant = true) : name_(name), is_constant_(is_constant), is_dangling_(is_dangling), types_(type.GetTypes()) {
    SetTensorProto(type.GetDefaultType());
    for (auto dim : shape) {
      t_proto_.add_dims(dim);
    }
  }

  IArg(const std::string& name, const PatternType& type, int rank,
       bool is_dangling = true, bool is_constant = true) : name_(name), is_constant_(is_constant), is_dangling_(is_dangling), types_(type.GetTypes()) {
    SetTensorProto(type.GetDefaultType());
    while (rank--) {
      t_proto_.add_dims(1);
    }
  }

  NodeDef GetNodeDef() const {
    return NodeDef("Constant",
                   {},
                   {ArgDef(name_, nullptr)},
                   {ONNX_NAMESPACE::MakeAttribute("value", t_proto_)});
  }

  std::string GetArgName() const {
    return name_;
  }

  bool IsSupportedType(const Graph& graph, const NodeArg& input_arg) const {
    const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
    if (is_constant_) {
      tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
    } else if (!graph.GetInitializedTensor(input_arg.Name(), tensor_proto)) {
      return false;
    }
    const auto data_type = tensor_proto->data_type();
    return std::find(types_.begin(), types_.end(), data_type) != types_.end();
  }

  bool IsExpectedShape(const Graph& graph, const NodeArg& input_arg) const {
    const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
    if (is_constant_) {
      tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
    } else if (!graph.GetInitializedTensor(input_arg.Name(), tensor_proto)) {
      return false;
    }
    if (t_proto_.dims_size() != tensor_proto->dims_size()) {
      return false;
    }
    auto pdims = t_proto_.dims();
    auto tdims = tensor_proto->dims();
    for (int i = 0; i < t_proto_.dims_size(); i++) {
      if (pdims[i] != tdims[i]) {
        return false;
      }
    }
    return true;
  }

  bool IsConstant() const {
    return is_constant_;
  }

  bool IsDangling() const {
    return is_dangling_;
  }

 private:
  void SetTensorProto(ONNX_NAMESPACE::TensorProto_DataType type) {
    switch (type) {
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
        t_proto_ = ONNX_NAMESPACE::ToTensor<int32_t>(0);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
        t_proto_ = ONNX_NAMESPACE::ToTensor<int64_t>(0);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16:
        t_proto_ = ONNX_NAMESPACE::ToTensor<int32_t>(0);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
        t_proto_ = ONNX_NAMESPACE::ToTensor<int32_t>(0);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL:
        t_proto_ = ONNX_NAMESPACE::ToTensor<int32_t>(0);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
        t_proto_ = ONNX_NAMESPACE::ToTensor<uint64_t>(0);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32:
        t_proto_ = ONNX_NAMESPACE::ToTensor<uint64_t>(0);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64:
        t_proto_ = ONNX_NAMESPACE::ToTensor<uint64_t>(0);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
        t_proto_ = ONNX_NAMESPACE::ToTensor<uint64_t>(0);
        break;
      default:
        t_proto_ = ONNX_NAMESPACE::ToTensor(.0, type);
        break;
    }
  }

 private:
  std::string name_;
  bool is_constant_;
  bool is_dangling_;
  std::vector<ONNX_NAMESPACE::TensorProto_DataType> types_;
  TensorProto t_proto_;
};

struct PNode {
  PNode(const std::string& op_type,
        const std::vector<std::string>& input_args_name,
        const std::vector<std::string>& output_args_name,
        const std::string& node_name = "",
        const std::vector<std::string>& domains = {},
        const std::vector<int>& versions = {},
        const std::vector<AttributeProto>& attributes = {})
      : op_type_(op_type),
        input_args_name(input_args_name),
        output_args_name(output_args_name),
        node_name_(node_name),
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
    valid_domains = std::unordered_set<std::string>(domains.begin(), domains.end());
    valid_versions = std::unordered_set<int>(versions.begin(), versions.end());
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
    if (domains.empty()) {
      return true;
    }
    return std::find(domains.begin(), domains.end(), domain) != domains.end();
  }

  bool VersionsContain(int version) const {
    if (versions.empty()) {
      return true;
    }
    return std::find(versions.begin(), versions.end(), version) != versions.end();
  }

  std::unordered_set<std::string> GetValidDomains() const { return valid_domains; }

  std::unordered_set<int> GetValidVersions() const { return valid_versions; }

  NodeDef GetNodeDef() const {
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
  std::vector<std::string> input_args_name;
  std::vector<std::string> output_args_name;
  std::string node_name_;
  std::vector<std::string> domains;
  std::vector<int> versions;
  std::vector<AttributeProto> attributes;
  std::unordered_set<std::string> valid_domains;
  std::unordered_set<int> valid_versions;
};

struct PatternGraph;

class NodeCompareFunc {
 public:
  virtual bool operator()(const Node* g, const PNode* p, const Graph& /*target*/, const PatternGraph& /*pattern*/) const = 0;
};

class ArgCompareFunc {
 public:
  virtual bool operator()(const NodeArg* g_arg, const IArg* p_arg, const Graph& /*target*/, const PatternGraph& /*pattern*/) const = 0;
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
    if (!skip_optype_ && !p->OpTypeEquals(g->OpType())) {
      PARSER_LOG << "OpType mismatch, "
                 << "g is: " << g->OpType();
      return false;
    }
    if (!skip_domain_ && !p->DomainsContain(g->Domain())) {
      PARSER_LOG << "Domain mismatch, "
                 << "g is: " << g->Domain();
      return false;
    }
    if (!skip_version_ && !p->VersionsContain(g->SinceVersion())) {
      PARSER_LOG << "Version mismatch, "
                 << "g is: " << g->SinceVersion();
      return false;
    }
    return true;
  }

  bool skip_optype_, skip_domain_, skip_version_, skip_path_;
};

class DefaultArgCompareFunc : public ArgCompareFunc {
 public:
  bool operator()(const NodeArg* g_arg, const IArg* p_arg, const Graph& target, const PatternGraph& /*pattern*/) const override {
    return p_arg->IsDangling() || p_arg->IsSupportedType(target, *g_arg);
  }
};

struct PatternGraph {
  PatternGraph(const std::vector<IArg>& constants, const std::vector<PNode>& pnodes,
               const std::string& name = "")
      : name_(name),
        pargs(constants),
        pnodes(pnodes) {
    ort_model_ptr_ = std::make_unique<Model>("PatternModel", false, logging::LoggingManager::DefaultLogger());
    for (size_t i = 0; i < pnodes.size(); i++) {
      nodes.push_back(pnodes[i].GetNodeDef());
      name_pnode_mapping_[pnodes[i].GetNodeName()] = &(this->pnodes[i]);
      domain_map[pnodes[i].GetNodeName()] = pnodes[i].GetValidDomains();
      version_map[pnodes[i].GetNodeName()] = pnodes[i].GetValidVersions();
    }
    for (size_t i = 0; i < constants.size(); i++) {
      nodes.push_back(constants[i].GetNodeDef());
      name_parg_mapping_[pargs[i].GetArgName()] = &(this->pargs[i]);
    }

    ORT_ENFORCE(ToGraphInternal().IsOK());

    default_node_compare_func_ = std::make_unique<DefaultNodeCompareFunc>(false, false, false, false);
    default_arg_compare_func_ = std::make_unique<DefaultArgCompareFunc>();
  }

  const std::string& name() const {
    return name_;
  }

  PatternGraph& SetCustomConstraint(std::unique_ptr<NodeCompareFunc> func, std::string node_name = "") {
    if (node_name.empty()) {
      default_node_compare_func_ = std::move(func);
      custom_node_constraints_.clear();
    } else {
      custom_node_constraints_[node_name] = std::move(func);
    }
    return *this;
  }

  PatternGraph& SetCustomConstraint(std::unique_ptr<ArgCompareFunc> func, std::string arg_name = "") {
    if (arg_name.empty()) {
      default_arg_compare_func_ = std::move(func);
      custom_arg_constraints_.clear();
    } else {
      custom_arg_constraints_[arg_name] = std::move(func);
    }
    return *this;
  }

  Graph& GetGraph() {
    return ort_model_ptr_->MainGraph();
  }

  Status TryMatch(Graph& target_graph, PatternMatchResult& res, const std::string& root_node = "");

 private:
  Status ToGraphInternal() {
    // Todo: p_initializer_names_to_preserve should contains all names in the pattern graphs.
    const std::unordered_set<std::string>* p_initializer_names_to_preserve = nullptr;
    GraphAugmenter::GraphDefs graph_defs;
    graph_defs.AddNodeDefs(nodes);
    auto status = GraphAugmenter::AugmentGraph(ort_model_ptr_->MainGraph(), graph_defs, p_initializer_names_to_preserve);
    ORT_ENFORCE(status.IsOK());
    // ORT_ENFORCE(GraphAugmenter::AugmentGraph(ort_model_ptr_->MainGraph(), graph_defs, p_initializer_names_to_preserve).IsOK());

    return Status::OK();
  }

  bool FindMatchRecursively(const Node* g, const Node* p,
                            std::unordered_set<const Node*>& graph_path, std::unordered_set<const Node*>& pattern_path,
                            std::unordered_map<const Node*, const Node*>& path_map,
                            PatternMatchResult& matched, const Graph& target);

  std::string name_;  // name of the graph

  std::vector<IArg> pargs;     // arg definitions
  std::vector<NodeDef> nodes;  // node definitions
  std::vector<PNode> pnodes;   // node definitions
  std::unique_ptr<Model> ort_model_ptr_;
  std::map<std::string, std::unordered_set<std::string>> domain_map;
  std::map<std::string, std::unordered_set<int>> version_map;

  std::unordered_map<std::string, const PNode*> name_pnode_mapping_;
  std::unordered_map<std::string, const IArg*> name_parg_mapping_;

  std::unique_ptr<NodeCompareFunc> default_node_compare_func_;
  std::unordered_map<std::string, std::unique_ptr<NodeCompareFunc>> custom_node_constraints_;
  std::unique_ptr<ArgCompareFunc> default_arg_compare_func_;
  std::unordered_map<std::string, std::unique_ptr<ArgCompareFunc>> custom_arg_constraints_;
};

}  // namespace training
}  // namespace onnxruntime