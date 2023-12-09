#include "core/graph/graph_view_api_impl.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/graph_proto_serializer.h"
#include "onnx/onnx_pb.h"

namespace onnxruntime {
const onnx::TensorShapeProto* GetNodeArgShape(const NodeArg* node_arg) {
  if (node_arg == nullptr) {
    return nullptr;
  }

  const auto* type = node_arg->TypeAsProto();
  if (type == nullptr) {
    return nullptr;
  }

  return utils::TryGetShape(*type);
}

std::optional<std::vector<int64_t>> GetShapeAsVector(const NodeArg& node_arg) {
  const auto* shape_proto = GetNodeArgShape(&node_arg);
  if (shape_proto == nullptr) {
    return std::nullopt;
  }

  TensorShape shape = utils::GetTensorShapeFromTensorShapeProto(*shape_proto);
  const auto dims = shape.GetDims();
  std::vector<int64_t> result;
  result.reserve(dims.size());
  result.assign(dims.begin(), dims.end());
  return result;
}

onnxruntime::DataType GetValueInfoDataType(const NodeArg& node_arg) {
  const auto* type = node_arg.TypeAsProto();
  if (!type) {
    return onnxruntime::DataType::UNDEFINED;
  }

  if (!utils::HasTensorType(*type)) {
    return onnxruntime::DataType::UNDEFINED;
  }

  if (!utils::HasElementType(*type)) {
    return onnxruntime::DataType::UNDEFINED;
  }

  return gsl::narrow_cast<onnxruntime::DataType>(type->tensor_type().elem_type());
}

// <ApiValueInfoView>
std::string_view ApiValueInfoView::Name() const {
  return node_arg_.Name();
}

std::optional<std::vector<int64_t>> ApiValueInfoView::Shape() const {
  return GetShapeAsVector(node_arg_);
}

onnxruntime::DataType ApiValueInfoView::DType() const {
  return GetValueInfoDataType(node_arg_);
}

// <ApiTensor>
std::vector<int64_t> ApiTensor::Shape() const {
  TensorShape shape = utils::GetTensorShapeFromTensorProto(tensor_proto_);
  const auto dims = shape.GetDims();
  return std::vector<int64_t>{dims.begin(), dims.end()};
}

size_t ApiTensor::NumElements() const {
  int64_t size = utils::GetTensorShapeFromTensorProto(tensor_proto_).Size();
  ORT_ENFORCE(size >= 0, "Failed to get size of TensorProto");
  return gsl::narrow_cast<size_t>(size);
}

onnxruntime::DataType ApiTensor::DType() const {
  return gsl::narrow_cast<onnxruntime::DataType>(tensor_proto_.data_type());
}

std::vector<uint8_t> ApiTensor::Data() const {
  // Reading tensor values from tensor_proto requires some work because of external storage and special/raw_data fields
  const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto_.data_type())->GetElementType();
  auto tensor_shape_dims = utils::GetTensorShapeFromTensorProto(tensor_proto_);
  TensorShape tensor_shape{std::move(tensor_shape_dims)};
  onnxruntime::Tensor tensor(tensor_dtype, tensor_shape, cpu_allocator_);
  ORT_THROW_IF_ERROR(utils::TensorProtoToTensor(Env::Default(), model_path_.ToPathString().c_str(),
                                                tensor_proto_, tensor));
  size_t num_bytes = gsl::narrow_cast<size_t>(tensor.SizeInBytes());
  const uint8_t* data = static_cast<const uint8_t*>(tensor.DataRaw());
  return std::vector<uint8_t>(data, data + num_bytes);
}
// </ApiTensor>

std::vector<std::string_view> NodeArgsToStrings(ConstPointerContainer<std::vector<NodeArg*>> node_args) {
  std::vector<std::string_view> result;
  result.reserve(node_args.size());
  for (const auto* arg : node_args) {
    result.push_back(arg->Name());
  }

  return result;
}

// <ApiNodeView>
std::vector<std::string_view> ApiNodeView::Inputs() const {
  return NodeArgsToStrings(node_.InputDefs());
}

std::vector<std::string_view> ApiNodeView::Outputs() const {
  return NodeArgsToStrings(node_.OutputDefs());
}

std::optional<int64_t> ApiNodeView::GetAttributeInt(std::string_view name) const {
  const onnx::AttributeProto* attr = graph_utils::GetNodeAttribute(node_, std::string(name));
  if (attr == nullptr || attr->type() != onnx::AttributeProto_AttributeType_INT) {
    return std::nullopt;
  }

  return attr->i();
}

std::optional<std::string> ApiNodeView::GetAttributeString(std::string_view name) const {
  const onnx::AttributeProto* attr = graph_utils::GetNodeAttribute(node_, std::string(name));
  if (attr == nullptr || attr->type() != onnx::AttributeProto_AttributeType_STRING) {
    return std::nullopt;
  }

  return attr->s();
}

std::optional<std::vector<int64_t>> ApiNodeView::GetAttributeInts(std::string_view name) const {
  const onnx::AttributeProto* attr = graph_utils::GetNodeAttribute(node_, std::string(name));
  if (attr == nullptr || attr->type() != onnx::AttributeProto_AttributeType_INTS) {
    return std::nullopt;
  }

  std::vector<int64_t> value;
  const auto& ints = attr->ints();
  value.reserve(ints.size());
  for (int64_t x : ints) {
    value.push_back(x);
  }

  return value;
}

std::optional<std::vector<float>> ApiNodeView::GetAttributeFloats(std::string_view name) const {
  const onnx::AttributeProto* attr = graph_utils::GetNodeAttribute(node_, std::string(name));
  if (attr == nullptr || attr->type() != onnx::AttributeProto_AttributeType_FLOATS) {
    return std::nullopt;
  }

  std::vector<float> value;
  const auto& floats = attr->floats();
  value.reserve(floats.size());
  for (float x : floats) {
    value.push_back(x);
  }

  return value;
}

void ApiNodeView::ForEachDef(std::function<void(const interface::ValueInfoViewRef&, bool is_input)> func, bool include_missing_optional_defs) const {
   for (const NodeArg* arg : node_.InputDefs()) {
    if (include_missing_optional_defs || arg->Exists()) {
      ApiValueInfoView value_view(*arg);
      func(value_view, true);
    }
   }
   for (const NodeArg* arg : node_.ImplicitInputDefs()) {
    if (include_missing_optional_defs || arg->Exists()) {
      ApiValueInfoView value_view(*arg);
      func(value_view, true);
    }
   }
   for (const NodeArg* arg : node_.OutputDefs()) {
    if (include_missing_optional_defs || arg->Exists()) {
      ApiValueInfoView value_view(*arg);
      func(value_view, false);
    }
   }
}

int ApiNodeView::SinceVersion() const {
  return node_.SinceVersion();
}

std::vector<std::unique_ptr<interface::GraphViewRef>> ApiNodeView::GetSubgraphs() const {
  std::vector<std::unique_ptr<interface::GraphViewRef>> ret;
  for (const auto& sub_graph : node_.GetSubgraphs()) {
    AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
    std::unique_ptr<ApiGraphView> graph_view = std::make_unique<ApiGraphView>(*sub_graph, std::move(cpu_allocator));
    ret.emplace_back(std::move(graph_view));
  }
  return ret;
}
// </ApiNodeView>

std::unique_ptr<interface::TensorRef> CreateApiTensor(const onnx::TensorProto* tensor, const Path& path, AllocatorPtr cpu_allocator) {
  if (tensor == nullptr) return nullptr;
  return std::make_unique<ApiTensor>(*tensor, path, std::move(cpu_allocator));
}


// <ApiGraphView>
std::string_view ApiGraphView::Name() const {
  if (isg_) return isg_->GetMetaDef()->name;
  return graph_.Name();
}

std::basic_string_view<ORTCHAR_T> ApiGraphView::ModelPath() const {
  return graph_.ModelPath().ToPathString();
}

std::optional<int64_t> ApiGraphView::Opset(std::string_view domain) const {
  const auto& version_map = graph_.DomainToVersionMap();
  auto match = version_map.find(std::string(domain));
  if (match == version_map.end()) {
    return std::nullopt;
  }

  return match->second;
}

std::vector<std::unique_ptr<interface::NodeViewRef>> ApiGraphView::NodeViews() const {
  GraphViewer graph_viewer(graph_, isg_);
  std::vector<std::unique_ptr<interface::NodeViewRef>> nodes;
  const auto& sorted_nodes = graph_viewer.GetNodesInTopologicalOrder();
  nodes.reserve(sorted_nodes.size());
  for (NodeIndex index : sorted_nodes) {
    auto& node = *graph_.GetNode(index);
    nodes.push_back(std::make_unique<ApiNodeView>(node));
  }

  return nodes;
}

std::unique_ptr<interface::TensorRef> ApiGraphView::GetConstant(std::string_view name) const {
  return CreateApiTensor(graph_.GetConstantInitializer(std::string(name), /*check_outer_scope*/ true), graph_.ModelPath(), cpu_allocator_);
}

std::unique_ptr<interface::NodeViewRef> ApiGraphView::GetNode(size_t node_index) const {
  GraphViewer graph_viewer(graph_, isg_);
  return std::make_unique<ApiNodeView>(*graph_viewer.GetNode(node_index));
}

std::vector<std::string_view> ApiGraphView::GetInputsIncludingInitializers() const {
  GraphViewer graph_viewer(graph_, isg_);
  const std::vector<const NodeArg*>& inputs_including_initializers = graph_viewer.GetInputsIncludingInitializers();
  std::vector<std::string_view> ret;
  ret.reserve(inputs_including_initializers.size());
  for (const NodeArg* input : inputs_including_initializers)
    ret.push_back(input->Name());
  return ret;
}

std::vector<std::string_view> ApiGraphView::GetInputs() const {
  GraphViewer graph_viewer(graph_, isg_);
  const std::vector<const NodeArg*>& node_args = graph_viewer.GetInputs();
  std::vector<std::string_view> ret;
  ret.reserve(node_args.size());
  for (const auto* arg : node_args) {
    ret.push_back(arg->Name());
  }

  return ret;
}

std::vector<std::string_view> ApiGraphView::GetOutputs() const {
  GraphViewer graph_viewer(graph_, isg_);
  const std::vector<const NodeArg*>& node_args = graph_viewer.GetOutputs();
  std::vector<std::string_view> ret;
  ret.reserve(node_args.size());
  for (const auto* arg : node_args) {
    ret.push_back(arg->Name());
  }

  return ret;
}

bool ApiGraphView::HasInitializerName(std::string_view name) const {
  GraphViewer graph_viewer(graph_, isg_); // TODO: make GraphViewer member variable
  return graph_viewer.GetAllInitializedTensors().count(std::string(name)) == 1;
}

bool ApiGraphView::IsConstantInitializer(std::string_view name, bool check_outer_scope) const {
  GraphViewer graph_viewer(graph_, isg_);
  return graph_viewer.IsConstantInitializer(std::string(name), check_outer_scope);
}

// TODO: return reference or value?
std::vector<size_t> ApiGraphView::GetNodesInTopologicalOrder() const {
  GraphViewer graph_viewer(graph_, isg_);
  return graph_viewer.GetNodesInTopologicalOrder();
}

std::unique_ptr<interface::ValueInfoViewRef> ApiGraphView::GetValueInfoView(std::string_view name) const {
  const NodeArg* node_arg_ = graph_.GetNodeArg(std::string(name));
  ORT_ENFORCE(node_arg_ != nullptr, "No NodeArg found for name ", name);
  return std::make_unique<ApiValueInfoView>(*node_arg_);
}

std::unique_ptr<interface::NodeViewRef> ApiGraphView::GetNodeViewProducingOutput(std::string_view name) const {
  const Node* producer = graph_.GetProducerNode(std::string(name));
  if (producer == nullptr) return nullptr;
  return std::make_unique<ApiNodeView>(*producer);
}

std::vector<std::unique_ptr<interface::NodeViewRef>> ApiGraphView::GetNodeViewsConsumingOutput(std::string_view name) const {
  std::vector<const Node*> consumers = graph_.GetConsumerNodes(std::string(name));
  std::vector<std::unique_ptr<interface::NodeViewRef>> ret;
  ret.reserve(consumers.size());
  for (const Node* node : consumers) {
    ret.push_back(std::make_unique<ApiNodeView>(*node));
  }
  return ret;
}

#ifdef INTREE_EP
onnx::ModelProto ApiGraphView::ToModelProto() const {
  GraphViewer graph_viewer(graph_, isg_);
  Model model(graph_viewer.Name(), true, ModelMetaData(), PathString(),
#if defined(ORT_MINIMAL_BUILD)
    IOnnxRuntimeOpSchemaRegistryList(),
#else
    IOnnxRuntimeOpSchemaRegistryList({graph_viewer.GetSchemaRegistry()}),
#endif
    graph_viewer.DomainToVersionMap(), std::vector<onnx::FunctionProto>(), graph_viewer.GetGraph().GetLogger()
  );
  onnx::ModelProto ret = model.ToProto();
  GraphViewerToProto(graph_viewer, *ret.mutable_graph(), true, true);
  return ret;
}
#endif

onnx::ModelProto* ApiGraphView::ToModelProto2() const {
  GraphViewer graph_viewer(graph_, isg_);
  Model model(graph_viewer.Name(), true, ModelMetaData(), PathString(),
#if defined(ORT_MINIMAL_BUILD)
    IOnnxRuntimeOpSchemaRegistryList(),
#else
    IOnnxRuntimeOpSchemaRegistryList({graph_viewer.GetSchemaRegistry()}),
#endif
    graph_viewer.DomainToVersionMap(), std::vector<onnx::FunctionProto>(), graph_viewer.GetGraph().GetLogger()
  );
  std::unique_ptr<onnx::ModelProto> ret = std::make_unique<onnx::ModelProto>(model.ToProto());
  GraphViewerToProto(graph_viewer, *(ret->mutable_graph()), true, true);
  return ret.release();
}

std::string_view ApiGraphView::SerializeModelProtoToString() const {
  GraphViewer graph_viewer(graph_, isg_);
  Model model(graph_viewer.Name(), true, ModelMetaData(), PathString(),
#if defined(ORT_MINIMAL_BUILD)
    IOnnxRuntimeOpSchemaRegistryList(),
#else
    IOnnxRuntimeOpSchemaRegistryList({graph_viewer.GetSchemaRegistry()}),
#endif
    graph_viewer.DomainToVersionMap(), std::vector<onnx::FunctionProto>(), graph_viewer.GetGraph().GetLogger()
  );
  onnx::ModelProto model_proto = model.ToProto();
  GraphViewerToProto(graph_viewer, *model_proto.mutable_graph(), true, true);
  std::string ret;
  model_proto.SerializeToString(&ret);
  return ret;
}

}
