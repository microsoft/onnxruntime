// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the provider DLL side of the bridge to let providers be built as a DLL
// It implements all of the unresolved externals and routes them across to the real functions in onnxruntime

#include "core/providers/dnnl/fake_proto.h"
//#include "core/framework/data_types.h"
//#include "core/framework/tensor.h"
//#include "core/framework/allocatormgr.h"
//#include "core/framework/execution_provider.h"

//#include "core/framework/kernel_def_builder.h"
//#include "core/graph/node_arg.h"
#include <assert.h>

onnxruntime::ProviderHost* g_host;

namespace onnxruntime {

void SetProviderHost(ProviderHost& host) {
  g_host = &host;
}
}  // namespace onnxruntime

// Override default new/delete so that we match the host's allocator
void* operator new(size_t n) { return g_host->HeapAllocate(n); }
void operator delete(void* p) { return g_host->HeapFree(p); }
void operator delete(void* p, size_t /*size*/) { return g_host->HeapFree(p); }

namespace onnx {
std::unique_ptr<ONNX_NAMESPACE::Prov_AttributeProto> Prov_AttributeProto::Create() {
  return g_host->AttributeProto_Create();
}
}  // namespace onnx

namespace onnxruntime {

Prov_AllocatorPtr CreateAllocator(Prov_DeviceAllocatorRegistrationInfo& info, int device_id) {
  return g_host->CreateAllocator(info, device_id);
}

std::unique_ptr<Prov_KernelDefBuilder> Prov_KernelDefBuilder::Create() {
  return g_host->KernelDefBuilder_Create();
}

std::shared_ptr<Prov_KernelRegistry> Prov_KernelRegistry::Create() {
  return g_host->KernelRegistry_Create();
}

std::unique_ptr<Prov_OrtMemoryInfo> Prov_OrtMemoryInfo::Create(const char* name_, OrtAllocatorType type_, Prov_OrtDevice* device_, int id_, OrtMemType mem_type_) {
  return g_host->OrtMemoryInfo_Create(name_, type_, device_, id_, mem_type_);
}

#if 0
	template <>
	MLDataType DataTypeImpl::GetType<bool>() {
		return nullptr;
	}
#endif

template <>
MLDataType DataTypeImpl::GetType<Tensor>() {
  return g_host->DataTypeImpl_GetType_Tensor();
}

template <>
MLDataType DataTypeImpl::GetType<float>() {
  return g_host->DataTypeImpl_GetType_float();
}

#if 0

	template <>
	MLDataType DataTypeImpl::GetType<double>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<uint8_t>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<int8_t>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<int16_t>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<uint16_t>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<int32_t>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<uint32_t>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<int64_t>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<uint64_t>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<BFloat16>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<MLFloat16>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<std::string>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::vector<std::map<int64_t, float>>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::vector<std::map<std::string, float>>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::map<int64_t, double>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::map<std::string, double>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::map<std::string, float>>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<std::map<std::string, int64_t>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::map<int64_t, float>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::map<int64_t, std::string>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetTensorType<bool>() { return nullptr; }
#endif

template <>
MLDataType DataTypeImpl::GetTensorType<float>() {
  return g_host->DataTypeImpl_GetTensorType_float();
}

#if 0
	template <>
	MLDataType DataTypeImpl::GetTensorType<double>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetTensorType<int8_t>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetTensorType<uint8_t>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetTensorType<int16_t>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetTensorType<uint16_t>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetTensorType<int32_t>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetTensorType<uint32_t>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetTensorType<int64_t>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetTensorType<uint64_t>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetTensorType<BFloat16>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetTensorType<MLFloat16>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetTensorType<std::string>() { return nullptr; }
#endif
}  // namespace onnxruntime

namespace onnx {

#if 0
int TensorShapeProto::dim_size() const {
  __debugbreak();
  return 0;
}
#endif

#if 0
::onnx::AttributeProto_AttributeType AttributeProto::type() const {
  __debugbreak();
  return ::onnx::AttributeProto_AttributeType_FLOAT;
}

int AttributeProto::ints_size() const {
  __debugbreak();
  return 0;
}

int64_t AttributeProto::ints(int i) const {
  __debugbreak();
  i;
  return 0;
}

int64_t AttributeProto::i() const {
  __debugbreak();
  return 0;
}

float AttributeProto::f() const {
  __debugbreak();
  return 0;
}

void AttributeProto::set_s(const ::std::string& value) {
  __debugbreak();
  value;
}

const ::std::string& AttributeProto::s() const {
  __debugbreak();
  static std::string s;
  return s;
}

void AttributeProto::set_name(const ::std::string& value) {
  __debugbreak();
  value;
}

void AttributeProto::set_type(::onnx::AttributeProto_AttributeType value) {
  __debugbreak();
  value;
}

::onnx::TensorProto* AttributeProto::add_tensors() {
  __debugbreak();
  return nullptr;
}
#endif

}  // namespace onnx

namespace onnxruntime {

void IndexedSubGraph::SetMetaDef(std::unique_ptr<MetaDef>& meta_def_) {
  __debugbreak();
  meta_def_;
}
#if 0
const std::string& NodeArg::Name() const noexcept {
  __debugbreak();
  static std::string s_string;
  return s_string;
}

const ONNX_NAMESPACE::TensorShapeProto* NodeArg::Shape() const {
  __debugbreak();
  return nullptr;
}

ONNX_NAMESPACE::DataType NodeArg::Type() const noexcept {
  __debugbreak();
  return nullptr;
}

const std::string& Node::OpType() const noexcept {
  __debugbreak();
  static std::string s_string;
  return s_string;
}

ConstPointerContainer<std::vector<NodeArg*>> Node::InputDefs() const noexcept {
  __debugbreak();
  return *(ConstPointerContainer<std::vector<NodeArg*>>*)nullptr;
}

ConstPointerContainer<std::vector<NodeArg*>> Node::OutputDefs() const noexcept {
  __debugbreak();
  return *(ConstPointerContainer<std::vector<NodeArg*>>*)nullptr;
}

NodeIndex Node::Index() const noexcept {
  __debugbreak();
  return 0;
}

const NodeAttributes& Node::GetAttributes() const noexcept {
  __debugbreak();
  return *(NodeAttributes*)nullptr;
}

size_t Node::GetInputEdgesCount() const noexcept {
  __debugbreak();
  return 0;
}

size_t Node::GetOutputEdgesCount() const noexcept {
  __debugbreak();
  return 0;
}

bool Node::NodeConstIterator::operator==(const NodeConstIterator& p_other) const {
  __debugbreak();
  p_other;
  return false;
}

bool Node::NodeConstIterator::operator!=(const NodeConstIterator& p_other) const {
  __debugbreak();
  p_other;
  return false;
}

void Node::NodeConstIterator::operator++() {
  __debugbreak();
}

void Node::NodeConstIterator::operator--() {
  __debugbreak();
}

const Node& Node::NodeConstIterator::operator*() const {
  __debugbreak();
  return *(Node*)nullptr;
}

Node::NodeConstIterator Node::InputNodesBegin() const noexcept {
  __debugbreak();
  return *(NodeConstIterator*)nullptr;
}

Node::NodeConstIterator Node::InputNodesEnd() const noexcept {
  __debugbreak();
  return *(NodeConstIterator*)nullptr;
}
#endif

#if 0
const std::string& GraphViewer::Name() const noexcept {
  __debugbreak();
  static std::string s_string;
  return s_string;
}

const Node* GraphViewer::GetNode(NodeIndex node_index) const {
  __debugbreak();
  node_index;
  return nullptr;
}

int GraphViewer::MaxNodeIndex() const noexcept {
  __debugbreak();
  return 0;
}

const InitializedTensorSet& GraphViewer::GetAllInitializedTensors() const noexcept {
  __debugbreak();
  return *(InitializedTensorSet*)nullptr;
}

const std::unordered_map<std::string, int>& GraphViewer::DomainToVersionMap() const noexcept {
  __debugbreak();
  return *(std::unordered_map<std::string, int>*)nullptr;
}
#endif

TensorShape::TensorShape() {
  __debugbreak();
}

TensorShape::TensorShape(const std::vector<int64_t>& dims) {
  __debugbreak();
  dims;
}

TensorShape::TensorShape(const std::initializer_list<int64_t>& dims) {
  __debugbreak();
  dims;
}

TensorShape::TensorShape(const int64_t* dimension_sizes, size_t dimension_count) {
  __debugbreak();
  dimension_sizes;
  dimension_count;
}

const int64_t& TensorShape::operator[](size_t idx) const {
  __debugbreak();
  idx;
  return *(int64_t*)nullptr;
}

int64_t& TensorShape::operator[](size_t idx) {
  __debugbreak();
  idx;

  return *(int64_t*)nullptr;
}

const std::vector<int64_t>& TensorShape::GetDims() const {
  __debugbreak();
  return *(std::vector<int64_t>*)nullptr;
}

int64_t TensorShape::Size() const {
  __debugbreak();
  return 0;
}

size_t TensorShape::NumDimensions() const noexcept {
  __debugbreak();
  return 0;
}

TensorShape TensorShape::Slice(size_t dimstart) const {
  __debugbreak();
  dimstart;
  return *(TensorShape*)nullptr;
}

std::string TensorShape::ToString() const {
  __debugbreak();
  return "";
}

const TensorShape& Tensor::Shape() const noexcept {
  __debugbreak();
  return *(TensorShape*)nullptr;
}

#if 0
KernelDefBuilder::KernelDefBuilder() {
  __debugbreak();
}

KernelDefBuilder& KernelDefBuilder::SetName(const char* op_name) {
  __debugbreak();
  op_name;
  return *this;
}
KernelDefBuilder& KernelDefBuilder::SetDomain(const char* domain) {
  __debugbreak();
  domain;
  return *this;
}
KernelDefBuilder& KernelDefBuilder::SinceVersion(int since_version) {
  __debugbreak();
  since_version;
  return *this;
}
KernelDefBuilder& KernelDefBuilder::Provider(const char* provider_type) {
  __debugbreak();
  provider_type;
  return *this;
}
KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* arg_name, MLDataType supported_type) {
  __debugbreak();
  arg_name;
  supported_type;
  return *this;
}

std::unique_ptr<KernelDef> KernelDefBuilder::Build() {
  __debugbreak();
  return nullptr;
}
#endif

Prov_ComputeCapability::Prov_ComputeCapability(std::unique_ptr<IndexedSubGraph> t_sub_graph) {
  __debugbreak();
  t_sub_graph;
}

OpKernel::OpKernel(const OpKernelInfo& info) {
  __debugbreak();
  info;
}

Tensor* OpKernelContext::Output(int index, const TensorShape& shape) {
  __debugbreak();
  index;
  shape;
  return nullptr;
}

#if 0
Prov_OrtMemoryInfo::Prov_OrtMemoryInfo(const char* name_, OrtAllocatorType type_, OrtDevice device_, int id_, OrtMemType mem_type_) {
  proxy_ = g_host->OrtMemoryInfo_constructor(name_, type_, device_, id_, mem_type_);
}
#endif

const CPUIDInfo& CPUIDInfo::GetCPUIDInfo() {
  __debugbreak();
  return *(CPUIDInfo*)nullptr;
}

bool CPUIDInfo::HasAVX2() const {
  __debugbreak();
  return false;
}

bool CPUIDInfo::HasAVX512f() const {
  __debugbreak();
  return false;
}

Prov_AllocatorPtr CreateAllocator(Prov_DeviceAllocatorRegistrationInfo info, int device_id) {
  return g_host->CreateAllocator(info, device_id);
}

std::unique_ptr<Prov_IDeviceAllocator> CreateCPUAllocator(std::unique_ptr<Prov_OrtMemoryInfo> info) {
  return g_host->CreateCPUAllocator(std::move(info));
}

Prov_AllocatorPtr CreateDummyArenaAllocator(Prov_AllocatorPtr resource_allocator) {
  __debugbreak();
  return nullptr;
}

#if 0
CPUAllocator::CPUAllocator(std::unique_ptr<OrtMemoryInfo> memory_info) {
  proxy_ = g_host->CPUAllocator_constructor(std::move(memory_info));
}

DummyArena::DummyArena(std::unique_ptr<IDeviceAllocator> resource_allocator) {
  __debugbreak();
  resource_allocator;
}
#endif

Prov_IExecutionProvider::Prov_IExecutionProvider(const std::string& type) {
  p_ = g_host->Create_IExecutionProvider_Router(this, type);
}

#if 0
Prov_IExecutionProvider::Prov_IExecutionProvider(const std::string& type) {
  proxy_ = g_host->IExecutionProvider_constructor(type);
}

IExecutionProvider::~IExecutionProvider() {
  g_host->IExecutionProvider_destructor(proxy_);
}

std::shared_ptr<KernelRegistry> IExecutionProvider::GetKernelRegistry() const {
  __debugbreak();
  return *(std::shared_ptr<KernelRegistry>*)nullptr;
}

std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                                                                  const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  __debugbreak();
  graph;
  return {};
}

common::Status IExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                           std::vector<NodeComputeInfo>& node_compute_funcs) {
  __debugbreak();
  fused_nodes;
  node_compute_funcs;
  return Status::OK();
}

Prov_AllocatorPtr IExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  __debugbreak();
  id;
  mem_type;
  return nullptr;
}

void IExecutionProvider::InsertAllocator(Prov_AllocatorPtr allocator) {
  g_host->IExecutionProvider_InsertAllocator(this, std::move(allocator));
}
#endif

namespace logging {

bool Logger::OutputIsEnabled(Severity severity, DataType data_type) const noexcept {
  severity;
  data_type;
  return false;
  // TODO: Logging not essential to make it work initially, do later
}

static Logger g_default_logger;

const Logger& LoggingManager::DefaultLogger() {
  return g_default_logger;
}

Capture::Capture(const Logger& logger, logging::Severity severity, const char* category,
                 logging::DataType dataType, const CodeLocation& location) {
  __debugbreak();
  logger;
  severity;
  category;
  dataType;
  location;
}

std::ostream& Capture::Stream() noexcept {
  __debugbreak();
  return *(std::ostream*)nullptr;
}

const char* Category::onnxruntime = "foo";

}  // namespace logging

namespace common {

Status::Status(StatusCategory category, int code, const std::string& msg) {
  __debugbreak();
  category;
  code;
  msg;
}

Status::Status(StatusCategory category, int code, const char* msg) {
  __debugbreak();
  category;
  code;
  msg;
}

std::string Status::ToString() const {
  __debugbreak();
  return "";
}

const std::string& Status::ErrorMessage() const noexcept {
  __debugbreak();
  static std::string dummy;
  return dummy;
}

}  // namespace common

std::vector<std::string> GetStackTrace() {
  __debugbreak();
  return {};
}

void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file, const char* function, uint32_t line) {
  return g_host->LogRuntimeError(session_id, status, file, function, line);
}

#if 0

const CPUIDInfo& CPUIDInfo::GetCPUIDInfo() {
  return g_host->CPUIDInfo_GetCPUIDInfo();
}

const OrtMemoryInfo& CPUAllocator::Info() const {
  return g_host->CPUAllocator_Info(this);
}

void* CPUAllocator::Alloc(uint64_t p1) {
  return g_host->CPUAllocator_Alloc(this, p1);
}
void CPUAllocator::Free(void* p1) {
  g_host->CPUAllocator_Free(this, p1);
}

std::shared_ptr<IAllocator> CreateAllocator(DeviceAllocatorRegistrationInfo info, int device_id) {
  return g_host->CreateAllocator(std::move(info), device_id);
}

Status IExecutionProvider::Compile(const std::vector<Node*>&, std::string&) {
  assert(false);
  return Status::OK();
}

Status IExecutionProvider::Compile(const std::vector<Node*>&, std::vector<NodeComputeInfo>&) {
  assert(false);
  return Status::OK();
}

void IExecutionProvider::InsertAllocator(std::shared_ptr<IAllocator> p1) {
  g_host->IExecutionProvider_InsertAllocator(this, std::move(p1));
}

Status IExecutionProvider::Sync() const {
  assert(false);
  return Status::OK();
}
Status IExecutionProvider::OnRunStart() {
  return g_host->IExecutionProvider_OnRunStart(this);
}

Status IExecutionProvider::OnRunEnd() {
  return g_host->IExecutionProvider_OnRunEnd(this);
}

std::shared_ptr<IAllocator> IExecutionProvider::GetAllocator(int p1, OrtMemType p2) const {
  return g_host->IExecutionProvider_GetAllocator(this, p1, p2);
}

std::shared_ptr<KernelRegistry> IExecutionProvider::GetKernelRegistry() const {
  assert(false);
  return nullptr;
}

std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider::GetCapability(const GraphViewer& p1, const std::vector<const KernelRegistry*>& p2) const {
  return g_host->IExecutionProvider_GetCapability(this, p1, p2);
}

Status KernelRegistry::Register(KernelCreateInfo&& p1) {
  return g_host->KernelRegistry_Register(this, std::move(p1));
}

std::ostream& operator<<(std::ostream& out, const DataTypeImpl* /*data_type*/) {
  assert(false);
  return out;
}

int64_t TensorShape::Size() const {
  return g_host->TensorShape_Size(this_);
}

TensorShape TensorShape::Slice(uint64_t p1) const {
  return g_host->TensorShape_Slice(this, p1);
}

KernelDefBuilder& KernelDefBuilder::Provider(char const* p1) {
  g_host->KernelDefBuilder_Provider(this_, p1);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::SetName(char const* p1) {
  g_host->KernelDefBuilder_SetName(this_, p1);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::SetDomain(char const* p1) {
  g_host->KernelDefBuilder_SetDomain(this_, p1);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(char const* p1, const DataTypeImpl* p2) {
  g_host->KernelDefBuilder_TypeConstraint(this_, p1, p2);
  return *this;
}

const NodeAttributes& Node::GetAttributes() const noexcept {
  return g_host->Node_GetAttributes(this);
}
NodeIndex Node::Index() const noexcept {
  return g_host->Node_Index(this);
}

const std::string& Node::OpType() const noexcept {
  return g_host->Node_OpType(this);
}

const ONNX_NAMESPACE::OpSchema* Node::Op() const noexcept {
  return g_host->Node_Op(this);
}

bool Node::NodeConstIterator::operator!=(const Node::NodeConstIterator& p1) const {
  return g_host->Node_NodeConstIterator_operator_not_equal(this, &p1);
}

void Node::NodeConstIterator::operator++() {
  return g_host->Node_NodeConstIterator_operator_plusplus(this);
}

const Node& Node::NodeConstIterator::operator*() const {
  return g_host->Node_NodeConstIterator_operator_star(this);
}

const std::string&
NodeArg::Name() const noexcept {
  return g_host->NodeArg_Name(this);
}

const ONNX_NAMESPACE::TensorShapeProto* NodeArg::Shape() const {
  return g_host->NodeArg_Shape(this);
}

ONNX_NAMESPACE::DataType NodeArg::Type() const noexcept {
  return g_host->NodeArg_Type(this);
}

int GraphViewer::MaxNodeIndex() const noexcept {
  return g_host->GraphViewer_MaxNodeIndex(this);
}

const std::string& GraphViewer::Name() const noexcept {
  return g_host->GraphViewer_Name(this);
}

const Node* GraphViewer::GetNode(NodeIndex p1) const {
  return g_host->GraphViewer_GetNode(this, p1);
}

const InitializedTensorSet& GraphViewer::GetAllInitializedTensors() const noexcept {
  return g_host->GraphViewer_GetAllInitializedTensors(this);
}

const Node& OpKernelInfo::node() const noexcept {
  assert(false);
  return *(Node*)nullptr;
}

Tensor* OpKernelContext::Output(int, const TensorShape&) {
  assert(false);
  return nullptr;
}

const OrtValue* OpKernelContext::GetInputMLValue(int) const {
  assert(false);
  return nullptr;
}

const IExecutionProvider* OpKernelInfo::GetExecutionProvider() const noexcept {
  assert(false);
  return nullptr;
}

Status OpKernelContext::GetTempSpaceAllocator(std::shared_ptr<IAllocator>*) const {
  assert(false);
  return Status::OK();
}

template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttr<float>(const std::string&, float*) const {
  assert(false);
  return Status::OK();
}

template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttr<int64_t>(const std::string&, int64_t*) const {
  assert(false);
  return Status::OK();
}

template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttr<std::string>(const std::string&, std::string*) const {
  assert(false);
  return Status::OK();
}

template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttrs<std::string>(const std::string&, std::vector<std::string>&) const {
  assert(false);
  return Status::OK();
}
template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttrs<int64_t>(const std::string&, std::vector<int64_t>&) const {
  assert(false);
  return Status::OK();
}

const KernelDef& OpKernelInfo::GetKernelDef() const {
  assert(false);
  return *(KernelDef*)nullptr;
}
#endif
}  // namespace onnxruntime
