// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {

#if BUILD_QNN_EP_STATIC_LIB
static std::unique_ptr<std::vector<std::function<void()>>> s_run_on_unload_;

void RunOnUnload(std::function<void()> function) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> guard(mutex);
  if (!s_run_on_unload_) {
    s_run_on_unload_ = std::make_unique<std::vector<std::function<void()>>>();
  }
  s_run_on_unload_->push_back(std::move(function));
}

struct OnUnload {
  ~OnUnload() {
    if (!s_run_on_unload_)
      return;

    for (auto& function : *s_run_on_unload_)
      function();

    s_run_on_unload_.reset();
  }

} g_on_unload;
#endif  // BUILD_QNN_EP_STATIC_LIB

void InitOrtCppApi() {
#if BUILD_QNN_EP_STATIC_LIB
  // Do nothing. Including "onnxruntime_cxx_api.h" normally initializes the global api_ object.
#else
  // Call util function in provider bridge that initializes the global api_ object.
  InitProviderOrtApi();
#endif
}

const ConfigOptions& RunOptions__GetConfigOptions(const RunOptions& run_options) {
#if BUILD_QNN_EP_STATIC_LIB
  return run_options.config_options;
#else
  return run_options.GetConfigOptions();
#endif
}

std::unique_ptr<IndexedSubGraph>& ComputeCapability__SubGraph(ComputeCapability& compute_cability) {
#if BUILD_QNN_EP_STATIC_LIB
  return compute_cability.sub_graph;
#else
  return compute_cability.SubGraph();
#endif
}

std::vector<NodeIndex>& IndexedSubGraph__Nodes(IndexedSubGraph& indexed_sub_graph) {
#if BUILD_QNN_EP_STATIC_LIB
  return indexed_sub_graph.nodes;
#else
  return indexed_sub_graph.Nodes();
#endif
}

std::vector<const Node*> Graph__Nodes(const Graph& graph) {
#if BUILD_QNN_EP_STATIC_LIB
  std::vector<const Node*> nodes;
  nodes.reserve(graph.NumberOfNodes());

  for (const Node& node : graph.Nodes()) {
    nodes.push_back(&node);
  }

  return nodes;
#else
  return graph.Nodes();
#endif
}

std::unique_ptr<Model> Model__Create(const std::string& graph_name, bool is_onnx_domain_only, const logging::Logger& logger) {
#if BUILD_QNN_EP_STATIC_LIB
  return std::make_unique<Model>(graph_name, is_onnx_domain_only, logger);
#else
  return Model::Create(graph_name, is_onnx_domain_only, logger);
#endif
}

std::unique_ptr<ModelMetadefIdGenerator> ModelMetadefIdGenerator__Create() {
#if BUILD_QNN_EP_STATIC_LIB
  return std::make_unique<ModelMetadefIdGenerator>();
#else
  return ModelMetadefIdGenerator::Create();
#endif
}

std::unique_ptr<ONNX_NAMESPACE::TypeProto> TypeProto__Create() {
#if BUILD_QNN_EP_STATIC_LIB
  return std::make_unique<ONNX_NAMESPACE::TypeProto>();
#else
  return ONNX_NAMESPACE::TypeProto::Create();
#endif
}

std::unique_ptr<Node_EdgeEnd> Node_EdgeEnd__Create(const Node& node, int src_arg_index, int dst_arg_index) {
#if BUILD_QNN_EP_STATIC_LIB
  return std::make_unique<Node_EdgeEnd>(node, src_arg_index, dst_arg_index);
#else
  return Node_EdgeEnd::Create(node, src_arg_index, dst_arg_index);
#endif
}

std::unique_ptr<NodeUnit> NodeUnit__Create(gsl::span<const Node* const> dq_nodes,
                                           const Node& target_node,
                                           gsl::span<const Node* const> q_nodes,
                                           NodeUnit::Type unit_type,
                                           gsl::span<const NodeUnitIODef> inputs,
                                           gsl::span<const NodeUnitIODef> outputs,
                                           size_t input_edge_count,
                                           gsl::span<const Node_EdgeEnd* const> output_edges) {
#if BUILD_QNN_EP_STATIC_LIB
  Node::EdgeSet output_edge_set;
  for (const Node_EdgeEnd* edge_end : output_edges) {
    output_edge_set.insert(*edge_end);
  }

  return std::make_unique<NodeUnit>(dq_nodes, target_node, q_nodes, unit_type,
                                    inputs, outputs, input_edge_count, output_edge_set);
#else
  return NodeUnit::Create(dq_nodes, target_node, q_nodes, unit_type, inputs, outputs, input_edge_count, output_edges);
#endif
}

std::pair<std::vector<std::unique_ptr<NodeUnit>>, std::unordered_map<const Node*, const NodeUnit*>>
GetQDQNodeUnits(const GraphViewer& graph_viewer, const logging::Logger& logger) {
#if BUILD_QNN_EP_STATIC_LIB
  return QDQ::GetAllNodeUnits(graph_viewer, logger);
#else
  return QDQ::GetAllNodeUnits(&graph_viewer, logger);
#endif
}

#if BUILD_QNN_EP_STATIC_LIB
#define NODE_ATTR_ITER_VAL(iter) (iter)->second
#else
#define NODE_ATTR_ITER_VAL(iter) (iter)->second()
#endif

NodeAttrHelper::NodeAttrHelper(const onnxruntime::Node& node)
    : node_attributes_(node.GetAttributes()) {}

NodeAttrHelper::NodeAttrHelper(const NodeUnit& node_unit)
    : node_attributes_(node_unit.GetNode().GetAttributes()) {}

float NodeAttrHelper::Get(const std::string& key, float def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return NODE_ATTR_ITER_VAL(entry).f();
  }

  return def_val;
}

int32_t NodeAttrHelper::Get(const std::string& key, int32_t def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return narrow<int32_t>(NODE_ATTR_ITER_VAL(entry).i());
  }

  return def_val;
}

uint32_t NodeAttrHelper::Get(const std::string& key, uint32_t def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return narrow<uint32_t>(NODE_ATTR_ITER_VAL(entry).i());
  }

  return def_val;
}

int64_t NodeAttrHelper::Get(const std::string& key, int64_t def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return NODE_ATTR_ITER_VAL(entry).i();
  }

  return def_val;
}

const std::string& NodeAttrHelper::Get(const std::string& key, const std::string& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return NODE_ATTR_ITER_VAL(entry).s();
  }

  return def_val;
}

std::vector<int32_t> NodeAttrHelper::Get(const std::string& key, const std::vector<int32_t>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = NODE_ATTR_ITER_VAL(entry).ints();
    const int64_t* cbegin = values.data();
    const int64_t* cend = values.data() + values.size();
    std::vector<int32_t> v;
    v.reserve(static_cast<size_t>(values.size()));
    std::transform(cbegin, cend, std::back_inserter(v),
                   [](int64_t val) -> int32_t { return narrow<int32_t>(val); });
    return v;
  }

  return def_val;
}

std::vector<uint32_t> NodeAttrHelper::Get(const std::string& key, const std::vector<uint32_t>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = NODE_ATTR_ITER_VAL(entry).ints();
    const int64_t* cbegin = values.data();
    const int64_t* cend = values.data() + values.size();
    std::vector<uint32_t> v;
    v.reserve(static_cast<size_t>(values.size()));
    std::transform(cbegin, cend, std::back_inserter(v),
                   [](int64_t val) -> uint32_t { return narrow<uint32_t>(val); });
    return v;
  }

  return def_val;
}

std::vector<int64_t> NodeAttrHelper::Get(const std::string& key, const std::vector<int64_t>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = NODE_ATTR_ITER_VAL(entry).ints();
    const int64_t* cbegin = values.data();
    const int64_t* cend = values.data() + values.size();
    return std::vector<int64_t>{cbegin, cend};
  }

  return def_val;
}

std::vector<float> NodeAttrHelper::Get(const std::string& key, const std::vector<float>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = NODE_ATTR_ITER_VAL(entry).floats();
    const float* cbegin = values.data();
    const float* cend = values.data() + values.size();
    return std::vector<float>{cbegin, cend};
  }

  return def_val;
}

std::optional<float> NodeAttrHelper::GetFloat(const std::string& key) const {
  std::optional<float> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    result = NODE_ATTR_ITER_VAL(entry).f();
  }

  return result;
}

std::optional<int64_t> NodeAttrHelper::GetInt64(const std::string& key) const {
  std::optional<int64_t> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    result = NODE_ATTR_ITER_VAL(entry).i();
  }

  return result;
}

std::optional<std::vector<float>> NodeAttrHelper::GetFloats(const std::string& key) const {
  std::optional<std::vector<float>> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = NODE_ATTR_ITER_VAL(entry).floats();
    const float* cbegin = values.data();
    const float* cend = values.data() + values.size();
    result = std::vector<float>(cbegin, cend);
  }

  return result;
}

std::optional<std::vector<int64_t>> NodeAttrHelper::GetInt64s(const std::string& key) const {
  std::optional<std::vector<int64_t>> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = NODE_ATTR_ITER_VAL(entry).ints();
    const int64_t* cbegin = values.data();
    const int64_t* cend = values.data() + values.size();
    result = std::vector<int64_t>(cbegin, cend);
  }

  return result;
}

std::optional<std::string> NodeAttrHelper::GetString(const std::string& key) const {
  std::optional<std::string> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    result = NODE_ATTR_ITER_VAL(entry).s();
  }

  return result;
}

bool NodeAttrHelper::HasAttr(const std::string& key) const {
  return node_attributes_.find(key) != node_attributes_.end();
}

namespace logging {
std::unique_ptr<Capture> Capture__Create(const Logger& logger, logging::Severity severity, const char* category,
                                         logging::DataType data_type, const CodeLocation& location) {
#if BUILD_QNN_EP_STATIC_LIB
  return std::make_unique<Capture>(logger, severity, category, data_type, location);
#else
  return Capture::Create(logger, severity, category, data_type, location);
#endif
}
}  // namespace logging
}  // namespace onnxruntime
