// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the provider DLL side of the bridge to let providers be built as a DLL
// It implements all of the unresolved externals and routes them across to the real functions in onnxruntime

#include "bridge_protobuf.h"
#include "core/framework/data_types.h"
#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"

#include "core/framework/kernel_def_builder.h"
#include "core/graph/node_arg.h"
#include "bridge.h"

onnxruntime::ProviderHost* g_host;

// Constructors/Destructors can't be routed across directly, so instead we create a special 'empty' version of the class in bridge_special.h
// that generates methods with the same signature as the real ones to keep the linker happy, and lets us then pass the 'this' pointer across
// to the real functions where we do a placement new or destructor call. The fake class must be empty and do nothing in order to not interfere
// with the real construction/destruction
void onnx_AttributeProto_constructor(void* _this) {
  g_host->onnx_AttributeProto_constructor(_this);
}

void onnx_AttributeProto_copy_constructor(void* _this, void* copy) {
  g_host->onnx_AttributeProto_copy_constructor(_this, copy);
}

void onnx_AttributeProto_destructor(void* _this) {
  g_host->onnx_AttributeProto_destructor(_this);
}

void onnxruntime_Status_constructor_1(void* _this, const void* category, int code, char const* msg) {
  g_host->onnxruntime_Status_constructor_1(_this, category, code, msg);
}

void onnxruntime_Status_constructor_2(void* _this, const void* category, int code, const void* std_string_msg) {
  g_host->onnxruntime_Status_constructor_2(_this, category, code, std_string_msg);
}

void onnxruntime_TensorShape_constructor(void* _this, int64_t const* p1, uint64_t p2) {
  g_host->onnxruntime_TensorShape_constructor(_this, p1, p2);
}

void onnxruntime_OpKernelInfo_constructor(void* _this, void* p1, void* p2, void* p3, void* p4, void* p5, void* p6, void* p7) {
  g_host->onnxruntime_OpKernelInfo_constructor(_this, p1, p2, p3, p4, p5, p6, p7);
}

void onnxruntime_OpKernelInfo_copy_constructor(void* _this, void* copy) {
  g_host->onnxruntime_OpKernelInfo_copy_constructor(_this, copy);
}

// Override default new/delete so that we match the host's allocator
void* operator new(size_t n) { return g_host->HeapAllocate(n); }
void operator delete(void* p) { return g_host->HeapFree(p); }

namespace onnxruntime {
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

#include "core/common/cpuid_info.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/op_kernel.h"
#include "core/framework/ort_value_tensor_slicer.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/node_arg.h"
#include "core/graph/graph_viewer.h"
#include "core/platform/threadpool.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnx {

void AttributeProto::CheckTypeAndMergeFrom(google::protobuf::MessageLite const&) { assert(false); }

void AttributeProto::CopyFrom(AttributeProto const& p1) { (this->*g_host->onnx_AttributeProto_CopyFrom)(p1); }

void AttributeProto::Clear() { assert(false); }

bool AttributeProto::IsInitialized() const {
  assert(false);
  return false;
}

uint64_t AttributeProto::ByteSizeLong() const {
  assert(false);
  return 0;
}

bool AttributeProto::MergePartialFromCodedStream(google::protobuf::io::CodedInputStream*) {
  assert(false);
  return false;
}

void AttributeProto::SerializeWithCachedSizes(google::protobuf::io::CodedOutputStream*) const { assert(false); }

bool AttributeProto_AttributeType_IsValid(int p1) {
  return g_host->onnx_AttributeProto_AttributeType_IsValid(p1);
}

std::string AttributeProto::GetTypeName() const {
  assert(false);
  return "";
}

void TensorProto::CopyFrom(TensorProto const& p1) { (this->*g_host->onnx_TensorProto_CopyFrom)(p1); }
}  // namespace onnx

google::protobuf::internal::LogMessage::LogMessage(google::protobuf::LogLevel, char const*, int) { assert(false); }
google::protobuf::internal::LogMessage::~LogMessage() { assert(false); }
google::protobuf::internal::LogMessage& google::protobuf::internal::LogMessage::operator<<(char const*) {
  assert(false);
  return *this;
}

void google::protobuf::internal::LogFinisher::operator=(google::protobuf::internal::LogMessage&) { assert(false); }

google::protobuf::MessageLite* google::protobuf::MessageLite::New(google::protobuf::Arena*) const {
  assert(false);
  return nullptr;
}

std::string google::protobuf::MessageLite::InitializationErrorString() const {
  assert(false);
  return "";
}

void google::protobuf::MessageLite::SerializeWithCachedSizes(google::protobuf::io::CodedOutputStream*) const { assert(false); }
unsigned char* google::protobuf::MessageLite::SerializeWithCachedSizesToArray(unsigned char*) const {
  assert(false);
  return nullptr;
}

unsigned char* google::protobuf::MessageLite::InternalSerializeWithCachedSizesToArray(bool, unsigned char*) const {
  assert(false);
  return nullptr;
}

void google::protobuf::internal::RepeatedPtrFieldBase::Reserve(int p1) {
  (this->*g_host->google_protobuf_internal_RepeatedPtrFieldBase_Reserve)(p1);
}

template <>
onnx::AttributeProto* google::protobuf::Arena::CreateMaybeMessage<onnx::AttributeProto>(google::protobuf::Arena*) {
  assert(false);
  return nullptr;
}
template <>
onnx::TensorProto* google::protobuf::Arena::CreateMaybeMessage<onnx::TensorProto>(google::protobuf::Arena* p1) {
  return g_host->google_protobuf_Arena_CreateMaybeMessage_onnx_TensorProto(p1);
}

const ::std::string& google::protobuf::internal::GetEmptyStringAlreadyInited() {
  return g_host->google_protobuf_internal_GetEmptyStringAlreadyInited();
}

namespace onnxruntime {

CPUIDInfo::CPUIDInfo() noexcept { assert(false); }

namespace logging {
Logger* LoggingManager::s_default_logger_{};
const char* Category::onnxruntime{};
Capture::~Capture() { assert(false); }

}  // namespace logging

namespace common {

std::string Status::ToString() const {
  assert(false);
  return "";
}

const std::string& Status::ErrorMessage() const noexcept {
  assert(false);
  static std::string dummy;
  return dummy;
}

}  // namespace common

std::vector<std::string> GetStackTrace() {
  assert(false);
  return {};
}

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
  return g_host->TensorShape_Size(this);
}

TensorShape TensorShape::Slice(uint64_t p1) const {
  return g_host->TensorShape_Slice(this, p1);
}

std::string TensorShape::ToString() const {
  assert(false);
  return "";
}

KernelDefBuilder& KernelDefBuilder::Provider(char const* p1) {
  return (this->*g_host->KernelDefBuilder_Provider)(p1);
}

KernelDefBuilder& KernelDefBuilder::SetName(char const* p1) {
  return (this->*g_host->KernelDefBuilder_SetName)(p1);
}

KernelDefBuilder& KernelDefBuilder::SetDomain(char const* p1) {
  return (this->*g_host->KernelDefBuilder_SetDomain)(p1);
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(char const* p1, const DataTypeImpl* p2) {
  return (this->*g_host->KernelDefBuilder_TypeConstraint)(p1, p2);
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

}  // namespace onnxruntime

#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/nn/pool.h"
#include "core/providers/cpu/nn/lrn.h"
#include "core/providers/cpu/nn/batch_norm.h"
#include "core/providers/cpu/nn/conv.h"

namespace onnxruntime {
template <>
Status Sum_6<float>::Compute(OpKernelContext*) const {
  assert(false);
  return Status::OK();
}

template <>
Status Pool<float, AveragePool>::Compute(OpKernelContext*) const {
  assert(false);
  return Status::OK();
}

template <>
Status Pool<float, MaxPool<1>>::Compute(OpKernelContext*) const {
  assert(false);
  return Status::OK();
}

template <>
Status Pool<float, MaxPool<8>>::Compute(OpKernelContext*) const {
  assert(false);
  return Status::OK();
}

template <>
Status LRN<float>::Compute(OpKernelContext*) const {
  assert(false);
  return Status::OK();
}

template <>
Status BatchNorm<float>::Compute(OpKernelContext*) const {
  assert(false);
  return Status::OK();
}

Status Conv<float>::Compute(OpKernelContext*) const {
  assert(false);
  return Status::OK();
}
}  // namespace onnxruntime
