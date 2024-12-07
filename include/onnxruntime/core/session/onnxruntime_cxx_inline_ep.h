// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Do not include this file directly. Please include "onnxruntime_cxx_api_ep.h" instead.

namespace Ort{
namespace PluginEP {

static const OrtGraphApi* ort_graph_api = GetApi().GetGraphApi(ORT_API_VERSION);

inline ValueInfoRef::ValueInfoRef(OrtValueInfoRef* value_info) : value_info_(value_info) {}

inline ValueInfoRef::~ValueInfoRef() {
  ort_graph_api->OrtGraph_ReleaseValueInfo(value_info_);
}

inline const std::vector<int64_t> ValueInfoRef::GetShape() {
  std::vector<int64_t> shape(value_info_->shape, value_info_->shape + value_info_->shape_len);
  return shape;
}

inline const ONNXTensorElementDataType ValueInfoRef::GetTensorElementType() {
  return value_info_->data_type;
}

inline Graph::Graph(const OrtGraphViewer* graph) : graph_(graph) {}

inline const char* Graph::GetName() {
  const char* graph_name = nullptr;
  ThrowOnError(ort_graph_api->OrtGraph_GetName(graph_, &graph_name));
  return graph_name;
}

inline bool Graph::IsConstantInitializer(const char* name, bool check_outer_scope) {
  bool is_initializer = false;
  ThrowOnError(ort_graph_api->OrtGraph_IsConstantInitializer(graph_, name, check_outer_scope, &is_initializer));
  return is_initializer;
}

inline const std::vector<size_t> Graph::GetNodesIndexInTopologicalOrder(int execution_order) {
  const size_t* nodes_index = nullptr;
  size_t nodes_count = 0;
  ThrowOnError(ort_graph_api->OrtGraph_GetNodesIndexInTopologicalOrder(graph_, execution_order, nodes_index, nodes_count));
  return std::vector<size_t>(nodes_index, nodes_index + nodes_count);
}

inline bool Graph::IsSubgraph() {
  bool is_subgraph = false;
  ThrowOnError(ort_graph_api->OrtGraph_IsSubgraph(graph_, &is_subgraph));
  return is_subgraph;
}

inline std::shared_ptr<Node> Graph::GetParenNode() {
  const OrtNode* parent_node = nullptr;
  ThrowOnError(ort_graph_api->OrtGraph_GetParenNode(graph_, &parent_node));
  return std::make_shared<Node>(parent_node);
}

inline std::filesystem::path Graph::GetModelPath() {
  const void* model_path = nullptr;
  ThrowOnError(ort_graph_api->OrtGraph_GetModelPath(graph_, &model_path));
  return *reinterpret_cast<const std::filesystem::path*>(model_path);
}

inline std::vector<std::string> Graph::GetRequiredInputs() {
  const char** required_inputs = nullptr;
  size_t required_inputs_count = 0;
  ThrowOnError(ort_graph_api->OrtGraph_GetRequiredInputs(graph_, &required_inputs, &required_inputs_count));
  auto release_fn = [](const char** strs) {
    ThrowOnError(ort_graph_api->ReleaseCharArray(strs));
  };
  std::unique_ptr<const char*, decltype(release_fn)> guard(required_inputs, release_fn);
  std::vector<std::string> ret;
  ret.reserve(required_inputs_count);
  for (size_t i = 0; i < required_inputs_count; i++) {
    ret.emplace_back(required_inputs[i]);
  }
  return ret;
}

inline std::vector<std::string> Graph::GetAllInputs() {
  const char** all_inputs = nullptr;
  size_t all_inputs_count = 0;
  ThrowOnError(ort_graph_api->OrtGraph_GetAllInputs(graph_, &all_inputs, &all_inputs_count));
  auto release_fn = [](const char** strs) {
    ThrowOnError(ort_graph_api->ReleaseCharArray(strs));
  };
  std::unique_ptr<const char*, decltype(release_fn)> guard(all_inputs, release_fn);
  std::vector<std::string> ret;
  ret.reserve(all_inputs_count);
  for (size_t i = 0; i < all_inputs_count; i++) {
    ret.emplace_back(all_inputs[i]);
  }
  return ret;
}

inline std::vector<std::string> Graph::GetAllInitializers() {
  const char** all_initializers = nullptr;
  size_t all_initializers_count = 0;
  ThrowOnError(ort_graph_api->OrtGraph_GetAllInitializers(graph_, &all_initializers, &all_initializers_count));
  auto release_fn = [](const char** strs) {
    ThrowOnError(ort_graph_api->ReleaseCharArray(strs));
  };
  std::unique_ptr<const char*, decltype(release_fn)> guard(all_initializers, release_fn);
  std::vector<std::string> ret;
  ret.reserve(all_initializers_count);
  for (size_t i = 0; i < all_initializers_count; i++) {
    ret.emplace_back(all_initializers[i]);
  }
  return ret;
}

inline Ort::PluginEP::Node Graph::GetOrtNode(size_t node_index) {
  const OrtNode* node = nullptr;
  ThrowOnError(ort_graph_api->OrtGraph_GetOrtNode(graph_, node_index, &node));
  return Ort::PluginEP::Node(node);
}

inline std::vector<Ort::PluginEP::Node> Graph::GetNodesConsumingInput(const char* input_name) {
  const OrtNode** consumers = nullptr;
  size_t consumer_count = 0;
  ThrowOnError(ort_graph_api->OrtGraph_GetNodesConsumingInput(graph_, input_name, &consumers, &consumer_count));
  std::vector<Ort::PluginEP::Node> ret;
  ret.reserve(consumer_count);
  for (size_t i = 0; i < consumer_count; i++) {
    ret.emplace_back(consumers[i]);
  }
  ort_graph_api->ReleaseOrtNodeArray(consumers);
  return ret;
}

inline Ort::PluginEP::Node Graph::GetNodeProducingOutput(const char* output_name) {
  const OrtNode* node = nullptr;
  ThrowOnError(ort_graph_api->OrtGraph_GetNodeProducingOutput(graph_, output_name, &node));
  return Ort::PluginEP::Node(node);
}

inline int Graph::NumberOfNodes() {
  int num_nodes = 0;
  ThrowOnError(ort_graph_api->OrtGraph_NumberOfNodes(graph_, &num_nodes));
  return num_nodes;
}

inline int Graph::MaxNodeIndex() {
  int max_node_index = 0;
  ThrowOnError(ort_graph_api->OrtGraph_MaxNodeIndex(graph_, &max_node_index));
  return max_node_index;
}

inline size_t Graph::GetOutputSize() {
  size_t output_size = 0;
  ThrowOnError(ort_graph_api->OrtGraph_GetOutputSize(graph_, &output_size));
  return output_size;
}

inline std::string Graph::GetIthOutputName(size_t i) {
  const char* output_name = nullptr;
  ThrowOnError(ort_graph_api->OrtGraph_GetIthOutputName(graph_, i, &output_name));
  return std::string(output_name);
}

inline int32_t Graph::GetIthOutputElemType(size_t i) {
  int32_t elem_type = 0;
  ThrowOnError(ort_graph_api->OrtGraph_GetIthOutputElemType(graph_, i, &elem_type));
  return elem_type;
}

inline std::shared_ptr<ValueInfoRef> Graph::GetValueInfo(const char* name) {
  OrtValueInfoRef* value_info = nullptr;
  ThrowOnError(ort_graph_api->OrtGraph_GetValueInfo(graph_, name, &value_info));
  return std::make_shared<ValueInfoRef>(value_info);
}

// inline void Graph::DumpOnnxModel(const std::filesystem::path& onnx_model_path) {
//   ThrowOnError(ort_graph_api->OrtGraph_DumpOnnxModel(graph_->GetGraph(), onnx_model_path.c_str()));
// }

inline std::shared_ptr<Graph> Graph::GetSubGraph(std::vector<size_t> node_indices) {
  const OrtGraphViewer* subgraph = nullptr;
  ThrowOnError(ort_graph_api->OrtGraph_GetSubGraph(graph_, node_indices.size(), node_indices.data(), &subgraph));
  // TODO:yang if should release subgraph in the decstructor of Graph?
  return std::make_shared<Graph>(subgraph);
}

// inline bool Graph::IsSameGraph(const Graph& other) {
//   bool is_same = false;
//   ThrowOnError(ort_graph_api->OrtGraph_IsSameGraph(graph_, other.GetGraph(), &is_same));
//   return is_same;
// }

inline Node::Node(const OrtNode* node) : node_(node) {}

inline const char* Node::GetName() {
  const char* node_name = nullptr;
  ThrowOnError(ort_graph_api->OrtNode_GetName(node_, &node_name));
  return node_name;
}

inline const std::string Node::GetDescription() {
  const char* node_description = nullptr;
  ThrowOnError(ort_graph_api->OrtNode_GetDescription(node_, &node_description));
  return std::string(node_description);
}

inline const std::string Node::GetDomain() {
  const char* node_domain = nullptr;
  ThrowOnError(ort_graph_api->OrtNode_GetDomain(node_, &node_domain));
  return std::string(node_domain);
}

inline int Node::SinceVersion() {
  int since_version = 0;
  ThrowOnError(ort_graph_api->OrtNode_SinceVersion(node_, &since_version));
  return since_version;
}

inline const std::string Node::GetExecutionProviderType() {
  const char* execution_provider_type = nullptr;
  ThrowOnError(ort_graph_api->OrtNode_GetExecutionProviderType(node_, &execution_provider_type));
  return std::string(execution_provider_type);
}

inline const std::string Node::GetOpType() {
  const char* op_type = nullptr;
  ThrowOnError(ort_graph_api->OrtNode_GetOpType(node_, &op_type));
  return std::string(op_type);
}

inline size_t Node::GetImplicitInputSize() {
  size_t implicit_input_size = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetImplicitInputSize(node_, &implicit_input_size));
  return implicit_input_size;
}

inline const std::string Node::GetIthImplicitInputName(size_t i) {
  const char* implicit_input_name = nullptr;
  ThrowOnError(ort_graph_api->OrtNode_GetIthImplicitInputName(node_, i, &implicit_input_name));
  return std::string(implicit_input_name);
}

inline size_t Node::GetNumInputs() {
  size_t num_inputs = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetNumInputs(node_, &num_inputs));
  return num_inputs;
}

inline const std::string Node::GetIthInputName(size_t i) {
  const char* input_name = nullptr;
  ThrowOnError(ort_graph_api->OrtNode_GetIthInputName(node_, i, &input_name));
  return std::string(input_name);
}

inline size_t Node::GetNumOutputs() {
  size_t num_outputs = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetNumOutputs(node_, &num_outputs));
  return num_outputs;
}

inline const std::string Node::GetIthOutputName(size_t i) {
  const char* output_name = nullptr;
  ThrowOnError(ort_graph_api->OrtNode_GetIthOutputName(node_, i, &output_name));
  return std::string(output_name);
}

inline size_t Node::GetIndex() {
  size_t node_index = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetIndex(node_, &node_index));
  return node_index;
}

inline std::vector<std::string> Node::GetAttributeNames() {
  const char** attribute_names = nullptr;
  size_t attribute_names_count = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeNames(node_, &attribute_names, &attribute_names_count));
  auto release_fn = [](const char** strs) {
    ThrowOnError(ort_graph_api->ReleaseCharArray(strs));
  };
  std::unique_ptr<const char*, decltype(release_fn)> guard(attribute_names, release_fn);
  std::vector<std::string> ret;
  ret.reserve(attribute_names_count);
  for (size_t i = 0; i < attribute_names_count; i++) {
    ret.emplace_back(attribute_names[i]);
  }
  return ret;
}

inline size_t Node::GetAttributeSize() {
  size_t attribute_size = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeSize(node_, &attribute_size));
  return attribute_size;
}

inline int Node::GetAttributeType(std::string attribute_name) {
  int attribute_type = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeType(node_, attribute_name.c_str(), &attribute_type));
  return attribute_type;
}

inline size_t Node::GetAttributeKeyCount(std::string attribute_name) {
  size_t attribute_key_count = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeKeyCount(node_, attribute_name.c_str(), &attribute_key_count));
  return attribute_key_count;
}

inline int Node::GetAttributeIntSize(std::string attribute_name) {
  int attribute_int_size = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeIntSize(node_, attribute_name.c_str(), &attribute_int_size));
  return attribute_int_size;
}

inline int Node::GetAttributeFloatSize(std::string attribute_name) {
  int attribute_float_size = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeFloatSize(node_, attribute_name.c_str(), &attribute_float_size));
  return attribute_float_size;
}

inline int Node::GetAttributeStringSize(std::string attribute_name) {
  int attribute_string_size = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeStringSize(node_, attribute_name.c_str(), &attribute_string_size));
  return attribute_string_size;
}

inline int64_t Node::GetAttributeIthInt(std::string attribute_name, size_t i) {
  int64_t attribute_ith_int = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeIthInt(node_, attribute_name.c_str(), i, &attribute_ith_int));
  return attribute_ith_int;
}

inline float Node::GetAttributeIthFloat(std::string attribute_name, size_t i) {
  float attribute_ith_float = 0.0f;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeIthFloat(node_, attribute_name.c_str(), i, &attribute_ith_float));
  return attribute_ith_float;
}

inline const std::string Node::GetAttributeIthStr(std::string attribute_name, size_t i) {
  const char* attribute_ith_string = nullptr;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeIthStr(node_, attribute_name.c_str(), i, &attribute_ith_string));
  return std::string(attribute_ith_string);
}

inline const std::string Node::GetAttributeStr(std::string attribute_name) {
  const char* attribute_str = nullptr;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeStr(node_, attribute_name.c_str(), &attribute_str));
  return std::string(attribute_str);
}

inline int64_t Node::GetAttributeInt(std::string attribute_name) {
  int64_t attribute_int = 0;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeInt(node_, attribute_name.c_str(), &attribute_int));
  return attribute_int;
}

inline float Node::GetAttributeFloat(std::string attribute_name) {
  float attribute_float = 0.0f;
  ThrowOnError(ort_graph_api->OrtNode_GetAttributeFloat(node_, attribute_name.c_str(), &attribute_float));
  return attribute_float;
}


}  // namespace Ort
}  // namespace PluginEP
