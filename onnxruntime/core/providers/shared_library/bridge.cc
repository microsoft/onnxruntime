#include "bridge_protobuf.h"
#include "core/framework/data_types.h"
#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_def_builder.h"
#include "core/graph/node_arg.h"
#include "bridge.h"

onnxruntime::ProviderHost* g_host;

// Constructors & Destructors must be handled
void onnx_AttributeProto_constructor(void* _this) {
  g_host->onnx_AttributeProto_constructor(_this);
}

void onnx_AttributeProto_copy_constructor(void* _this, void* copy) {
  g_host->onnx_AttributeProto_copy_constructor(_this, copy);
}

void onnx_AttributeProto_destructor(void* _this) {
  g_host->onnx_AttributeProto_destructor(_this);
}

void onnxruntime_TensorShape_constructor(void* _this, __int64 const* p1, unsigned __int64 p2) {
  g_host->onnxruntime_TensorShape_constructor(_this, p1, p2);
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
	MLDataType DataTypeImpl::GetType<std::vector<std::map<__int64, float>>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::vector<std::map<std::string, float>>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::map<__int64, double>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::map<std::string, double>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::map<std::string, float>>() {
		return nullptr;
	}

	template <>
	MLDataType DataTypeImpl::GetType<std::map<std::string, __int64>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::map<__int64, float>>() { return nullptr; }

	template <>
	MLDataType DataTypeImpl::GetType<std::map<__int64, std::string>>() { return nullptr; }

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

void AttributeProto::CheckTypeAndMergeFrom(google::protobuf::MessageLite const&) { __debugbreak(); }
void AttributeProto::CopyFrom(AttributeProto const& p1) { (this->*g_host->onnx_AttributeProto_CopyFrom)(p1); }
void AttributeProto::Clear() { __debugbreak(); }
bool AttributeProto::IsInitialized() const {
  __debugbreak();
  return false;
}
unsigned __int64 AttributeProto::ByteSizeLong() const {
  __debugbreak();
  return 0;
}
bool AttributeProto::MergePartialFromCodedStream(google::protobuf::io::CodedInputStream*) {
  __debugbreak();
  return false;
}
void AttributeProto::SerializeWithCachedSizes(google::protobuf::io::CodedOutputStream*) const { __debugbreak(); }
bool AttributeProto_AttributeType_IsValid(int p1) {
  return g_host->onnx_AttributeProto_AttributeType_IsValid(p1);
}
std::string AttributeProto::GetTypeName() const {
  __debugbreak();
  return "";
}

void TensorProto::CopyFrom(TensorProto const& p1) { (this->*g_host->onnx_TensorProto_CopyFrom)(p1); }
}  // namespace onnx

google::protobuf::internal::LogMessage::LogMessage(google::protobuf::LogLevel, char const*, int) { __debugbreak(); }
google::protobuf::internal::LogMessage::~LogMessage() { __debugbreak(); }
google::protobuf::internal::LogMessage& google::protobuf::internal::LogMessage::operator<<(char const*) {
  __debugbreak();
  return *this;
}

void google::protobuf::internal::LogFinisher::operator=(google::protobuf::internal::LogMessage&) { __debugbreak(); }

google::protobuf::MessageLite* google::protobuf::MessageLite::New(google::protobuf::Arena*) const {
  __debugbreak();
  return nullptr;
}

std::string google::protobuf::MessageLite::InitializationErrorString() const {
  __debugbreak();
  return "";
}

void google::protobuf::MessageLite::SerializeWithCachedSizes(google::protobuf::io::CodedOutputStream*) const { __debugbreak(); }
unsigned char* google::protobuf::MessageLite::SerializeWithCachedSizesToArray(unsigned char*) const {
  __debugbreak();
  return nullptr;
}

unsigned char* google::protobuf::MessageLite::InternalSerializeWithCachedSizesToArray(bool, unsigned char*) const {
  __debugbreak();
  return nullptr;
}

void google::protobuf::internal::RepeatedPtrFieldBase::Reserve(int p1) {
  (this->*g_host->google_protobuf_internal_RepeatedPtrFieldBase_Reserve)(p1);
}

template <>
onnx::AttributeProto* google::protobuf::Arena::CreateMaybeMessage<onnx::AttributeProto>(google::protobuf::Arena*) {
  __debugbreak();
  return nullptr;
}
template <>
onnx::TensorProto* google::protobuf::Arena::CreateMaybeMessage<onnx::TensorProto>(google::protobuf::Arena* p1) {
  return g_host->google_protobuf_Arena_CreateMaybeMessage_onnx_TensorProto(p1);
}

#if 0
google::protobuf::internal::ExplicitlyConstructed<std::string> google::protobuf::internal::fixed_address_empty_string() {
	return g_host->google::protobuf::internal::fixed_address_empty_string();
}
#endif

const ::std::string& google::protobuf::internal::GetEmptyStringAlreadyInited() {
  return g_host->google_protobuf_internal_GetEmptyStringAlreadyInited();
}

namespace onnxruntime {

CPUIDInfo::CPUIDInfo() noexcept { __debugbreak(); }

#if 0
	std::ostream& operator<<(std::ostream& stream, TensorShape const&) { return stream; }
#endif

namespace logging {
Logger* LoggingManager::s_default_logger_{};
const char* Category::onnxruntime{};
Capture::~Capture() { __debugbreak(); }

}  // namespace logging

namespace common {

Status::Status(StatusCategory, int, char const*) { __debugbreak(); }

Status::Status(StatusCategory category, int code, const std::string& msg) {
  __debugbreak();
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

#if 0
		Status
			MapNamesToMLValueIdxs(const std::vector<std::string>& names,
				const OrtValueNameIdxMap& ort_value_name_idx_map,
				std::vector<int>& ort_value_idxs) {
			return Status::OK();
		}
#endif

}  // namespace common

std::vector<std::string> GetStackTrace() {
  __debugbreak();
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
  __debugbreak();
  return Status::OK();
}

Status IExecutionProvider::Compile(const std::vector<Node*>&, std::vector<NodeComputeInfo>&) {
  __debugbreak();
  return Status::OK();
}

void IExecutionProvider::InsertAllocator(std::shared_ptr<IAllocator> p1) {
  g_host->IExecutionProvider_InsertAllocator(this, std::move(p1));
}

Status IExecutionProvider::Sync() const {
  __debugbreak();
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
  __debugbreak();
  return nullptr;
}

std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider::GetCapability(const GraphViewer& p1, const std::vector<const KernelRegistry*>& p2) const {
  return g_host->IExecutionProvider_GetCapability(this, p1, p2);
}

Status KernelRegistry::Register(KernelCreateInfo&& p1) {
  return g_host->KernelRegistry_Register(this, std::move(p1));
}

#if 0
	const std::vector<MLDataType>& DataTypeImpl::AllTensorTypes() {
		__debugbreak();
		static std::vector<MLDataType> temp;
		return temp;
	}
#endif

std::ostream& operator<<(std::ostream& out, const DataTypeImpl* data_type) {
  __debugbreak();
  return out;
}

int64_t TensorShape::Size() const {
  return g_host->TensorShape_Size(this);
}

#if 0
	int64_t TensorShape::SizeFromDimension(uint64_t) const { return 0; }
	int64_t TensorShape::SizeToDimension(uint64_t) const { return 0; }

#endif

TensorShape TensorShape::Slice(unsigned __int64 p1) const {
  return g_host->TensorShape_Slice(this, p1);
}

#if 0
	TensorShape TensorShape::Slice(uint64_t, uint64_t) const { return *(TensorShape*)nullptr; }
#endif

std::string TensorShape::ToString() const {
  __debugbreak();
  return "";
}

#if 0
	Status FeedsFetchesInfo::MapNamesToMLValueIdxs(const std::vector<std::string>&, const OrtValueNameIdxMap&, std::vector<int>&) { return Status::OK(); }

	Status FeedsFetchesManager::Create(const std::vector<std::string>&, std::vector<std::string> const&, OrtValueNameIdxMap const&, std::unique_ptr<FeedsFetchesManager>&) { return Status::OK(); }

#endif

KernelDefBuilder& KernelDefBuilder::Provider(char const* p1) {
  return g_host->KernelDefBuilder_Provider(this, p1);
}

KernelDefBuilder& KernelDefBuilder::SetName(char const* p1) {
  return g_host->KernelDefBuilder_SetName(this, p1);
}

KernelDefBuilder& KernelDefBuilder::SetDomain(char const* p1) {
  return g_host->KernelDefBuilder_SetDomain(this, p1);
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(char const* p1, const DataTypeImpl* p2) {
  return g_host->KernelDefBuilder_TypeConstraint(this, p1, p2);
}

#if 0
	KernelDefBuilder& KernelDefBuilder::TypeConstraint(char const*, const std::vector<const DataTypeImpl*>&) { return *this; }

	KernelDefBuilder& KernelDefBuilder::Alias(int, int) {
		return *this;
	}

	KernelDefBuilder& KernelDefBuilder::MayInplace(int, int) { return *this; }
#endif

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

#if 0

	const std::vector<const NodeArg*>& GraphViewer::GetInputs() const noexcept { return *(std::vector<const NodeArg*>*)nullptr; }

	const std::vector<const NodeArg*>& GraphViewer::GetOutputs() const noexcept {
		static std::vector<const NodeArg*> dummy;
		return dummy;
	}

	const std::vector<const NodeArg*>& GraphViewer::GetInputsIncludingInitializers() const noexcept {
		return *(std::vector<const NodeArg*>*)nullptr;
	}
#endif

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

#if 0

	const GraphViewer* SessionState::GetGraphViewer() const {
		return nullptr;
	}

	SessionState const* SessionState::GetSubgraphSessionState(uint64_t, std::string const&) const { return nullptr; }
#endif
OpKernelInfo::OpKernelInfo(const OpKernelInfo& other) : OpKernelInfo(other.node_, other.kernel_def_, *other.execution_provider_, other.constant_initialized_tensors_,
                                                                     other.ort_value_name_idx_map_, other.funcs_mgr_, other.data_transfer_mgr_) {
  __debugbreak();
}
OpKernelInfo::OpKernelInfo(const onnxruntime::Node& node,
                           const KernelDef& kernel_def,
                           const IExecutionProvider& execution_provider,
                           const std::unordered_map<int, OrtValue>& constant_initialized_tensors,
                           const OrtValueNameIdxMap& ort_value_name_idx_map,
                           const FuncManager& funcs_mgr,
                           const DataTransferManager& data_transfer_mgr)
    : OpNodeProtoHelper(&proto_helper_context_),
      node_(node),
      kernel_def_(kernel_def),
      execution_provider_(&execution_provider),
      constant_initialized_tensors_(constant_initialized_tensors),
      ort_value_name_idx_map_(ort_value_name_idx_map),
      funcs_mgr_(funcs_mgr),
      data_transfer_mgr_(data_transfer_mgr),
      proto_helper_context_(node) {
  __debugbreak();
}

const Node& OpKernelInfo::node() const noexcept {
  __debugbreak();
  return *(Node*)nullptr;
}

Tensor* OpKernelContext::Output(int, const TensorShape&) {
  __debugbreak();
  return nullptr;
}

#if 0

	const DataTypeImpl* OpKernelContext::InputType(int) const { return nullptr; }

	unsigned __int64 OpKernelContext::GetNodeIndex() const { return 0; }
#endif

const OrtValue* OpKernelContext::GetInputMLValue(int) const {
  __debugbreak();
  return nullptr;
}

#if 0

	OrtValue* OpKernelContext::GetOrCreateOutputMLValue(int) { return nullptr; }
#endif

const IExecutionProvider* OpKernelInfo::GetExecutionProvider() const noexcept {
  __debugbreak();
  return nullptr;
}

#if 0
	const OrtValue* OpKernelContext::GetImplicitInputMLValue(int) const {
		return nullptr;
	}

	OrtValue* OpKernelContext::GetOutputMLValue(int) {
		return nullptr;
	}

	int OpKernelContext::NumVariadicInputs(uint64_t) const { return 0; }
#endif
Status OpKernelContext::GetTempSpaceAllocator(std::shared_ptr<IAllocator>*) const {
  __debugbreak();
  return Status::OK();
}

#if 0
	uint64_t ProtoHelperNodeContext::getNumInputs() const { return 0; }
	uint64_t ProtoHelperNodeContext::getNumOutputs() const { return 0; }

	template <>
	template <>
	Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttr<onnx::GraphProto>(const std::string&, onnx::GraphProto*) const {
		return Status::OK();
	}
#endif

template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttr<float>(const std::string&, float*) const {
  __debugbreak();
  return Status::OK();
}

template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttr<int64_t>(const std::string&, int64_t*) const {
  __debugbreak();
  return Status::OK();
}

template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttr<std::string>(const std::string&, std::string*) const {
  __debugbreak();
  return Status::OK();
}

#if 0

	template <>
	template <>
	Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttr<ONNX_NAMESPACE::TensorProto>(const std::string&, ONNX_NAMESPACE::TensorProto*) const { return Status::OK(); }

	template <>
	template <>
	Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttrs<float>(const std::string&, std::vector<float>&) const {
		return Status::OK();
	}
#endif

template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttrs<std::string>(const std::string&, std::vector<std::string>&) const {
  __debugbreak();
  return Status::OK();
}
template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttrs<int64_t>(const std::string&, std::vector<int64_t>&) const {
  __debugbreak();
  return Status::OK();
}

#if 0
	template <>
	void OrtValueTensorSlicer<OrtValue>::Iterator::MaterializeMLValue() const {}

	template <>
	void OrtValueTensorSlicer<const OrtValue>::Iterator::MaterializeMLValue() const {}

	template <>
	OrtValueTensorSlicer<OrtValue> OrtValueTensorSlicer<OrtValue>::Create(OrtValue&, int64_t, int64_t) { return *(OrtValueTensorSlicer<OrtValue>*)nullptr; }

	template <>
	OrtValueTensorSlicer<const OrtValue> OrtValueTensorSlicer<const OrtValue>::Create(const OrtValue&, int64_t, int64_t) { return *(OrtValueTensorSlicer<const OrtValue>*)nullptr; }

	template <>
	OrtValueTensorSlicer<OrtValue>::Iterator::Iterator(OrtValue& ort_value, size_t, size_t, int64_t position, OrtValueTensorSlicer<OrtValue>::Iterator::Direction direction)
		: ort_value_{ &ort_value },
		position_{ position },
		increment_by_{ direction == Direction::kForward ? 1 : -1 },
		position_materialized_{ -1 } {
	}

	template <>
	OrtValueTensorSlicer<const OrtValue>::Iterator::Iterator(const OrtValue& ort_value, size_t, size_t, int64_t position, OrtValueTensorSlicer<const OrtValue>::Iterator::Direction direction)
		: ort_value_{ &ort_value },
		position_{ position },
		increment_by_{ direction == Direction::kForward ? 1 : -1 },
		position_materialized_{ -1 } {
	}

	namespace utils {

		Status ExecuteGraphWithCachedInfo(SessionState const&, FeedsFetchesManager const&, std::vector<OrtValue> const&, std::vector<OrtValue>&, std::unordered_map<unsigned __int64, std::function<Status(TensorShape const&, OrtValue&)>> const&, bool, bool const&, class onnxruntime::logging::Logger const&) { return Status::OK(); }

		Status ExecuteGraph(SessionState const&, FeedsFetchesManager&, std::vector<OrtValue> const&, std::vector<OrtValue>&, std::unordered_map<uint64_t, std::function<Status(TensorShape const&, OrtValue&)>> const&, bool, bool const&, logging::Logger const&, bool) { return Status::OK(); }

		TensorShape GetTensorShapeFromTensorShapeProto(ONNX_NAMESPACE::TensorShapeProto const&) {
			return *(TensorShape*)nullptr;
		}

		MLDataType
			GetMLDataType(NodeArg const&) {
			return nullptr;
		}

		enum ONNX_NAMESPACE::TensorProto_DataType GetTensorProtoType(const Tensor&) {
			return ONNX_NAMESPACE::TensorProto_DataType(0);
		}

		template <>
		Status UnpackTensor<bool>(const ONNX_NAMESPACE::TensorProto&, void const*, size_t, bool*, int64_t) { return Status::OK(); }
		template <>
		Status UnpackTensor<float>(const ONNX_NAMESPACE::TensorProto&, void const*, size_t, float*, int64_t) { return Status::OK(); }
		template <>
		Status UnpackTensor<double>(const ONNX_NAMESPACE::TensorProto&, void const*, size_t, double*, int64_t) { return Status::OK(); }
		template <>
		Status UnpackTensor<int8_t>(const ONNX_NAMESPACE::TensorProto&, void const*, size_t, int8_t*, int64_t) { return Status::OK(); }
		template <>
		Status UnpackTensor<uint8_t>(const ONNX_NAMESPACE::TensorProto&, void const*, size_t, uint8_t*, int64_t) { return Status::OK(); }
		template <>
		Status UnpackTensor<int16_t>(const ONNX_NAMESPACE::TensorProto&, void const*, size_t, int16_t*, int64_t) { return Status::OK(); }
		template <>
		Status UnpackTensor<uint16_t>(const ONNX_NAMESPACE::TensorProto&, void const*, size_t, uint16_t*, int64_t) { return Status::OK(); }
		template <>
		Status UnpackTensor<int32_t>(const ONNX_NAMESPACE::TensorProto&, void const*, size_t, int32_t*, int64_t) { return Status::OK(); }
		template <>
		Status UnpackTensor<uint32_t>(const ONNX_NAMESPACE::TensorProto&, void const*, size_t, uint32_t*, int64_t) { return Status::OK(); }
		template <>
		Status UnpackTensor<int64_t>(const ONNX_NAMESPACE::TensorProto&, void const*, size_t, int64_t*, int64_t) { return Status::OK(); }
		template <>
		Status UnpackTensor<uint64_t>(const ONNX_NAMESPACE::TensorProto&, void const*, size_t, uint64_t*, int64_t) { return Status::OK(); }
		template <>
		Status UnpackTensor<MLFloat16>(const ONNX_NAMESPACE::TensorProto&, void const*, size_t, MLFloat16*, int64_t) { return Status::OK(); }

	}  // namespace utils

	bool OpKernelInfo::TryGetConstantInput(int, const Tensor**) const {
		return false;
	}
#endif
const KernelDef& OpKernelInfo::GetKernelDef() const {
  __debugbreak();
  return *(KernelDef*)nullptr;
}

#if 0
	const std::vector<const DataTypeImpl*>& DataTypeImpl::AllFixedSizeTensorTypes() {
		static std::vector<const DataTypeImpl*> dummy;
		return dummy;
	}

	const std::vector<const DataTypeImpl*>& DataTypeImpl::AllNumericTensorTypes() {
		static std::vector<const DataTypeImpl*> dummy;
		return dummy;
	}

	Tensor::Tensor(DataTypeImpl const*, TensorShape const&, std::shared_ptr<IAllocator>, __int64) : alloc_info_(*(OrtMemoryInfo*)nullptr) {}
	Tensor::Tensor(DataTypeImpl const*, TensorShape const&, void*, OrtMemoryInfo const& info, __int64) : alloc_info_(info) {}
	Tensor::~Tensor() {}

	namespace math {

		float halfToFloat(uint16_t) { return 0.0f; }
		uint16_t floatToHalf(float) { return 0; }

		template <>
		void CopyVector<float, CPUMathUtil>(int, float const*, float*, CPUMathUtil*) {}

		template <>
		void RowwiseMax<float, CPUMathUtil>(int, int, float const*, float*, CPUMathUtil*) {}

		template <>
		void Gemm<float, CPUMathUtil>(CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, __int64, __int64, __int64, float, float const*, float const*, float, float*, CPUMathUtil*) {}
		template <>
		void GemmEx<float, CPUMathUtil>(CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, float, float const*, int, float const*, int, float, float*, int, CPUMathUtil*) {}

		template <>
		void Sqr<float, class CPUMathUtil>(int, float const*, float*, CPUMathUtil*) {}
		template <>
		void Axpy<float, CPUMathUtil>(int, float, float const*, float*, CPUMathUtil*) {}
		template <>
		void Powx<float, CPUMathUtil>(int, float const*, float, float*, CPUMathUtil*) {}
		template <>
		void Mul<float, CPUMathUtil>(int, float const*, float const*, float*, CPUMathUtil*) {}
		template <>
		void Col2im<float, CPUMathUtil, 2>(float const*, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float*, CPUMathUtil*) {}

		template <>
		void Gemv<float, CPUMathUtil>(enum CBLAS_TRANSPOSE, int, int, float, float const*, float const*, float, float*, CPUMathUtil*) {}

		template <>
		void Exp<float, CPUMathUtil>(int, float const*, float*, CPUMathUtil*) {}

		template <>
		void Set<float, CPUMathUtil>(__int64, float, float*, CPUMathUtil*) {}
	}  // namespace math

	namespace concurrency {

		//ThreadPool::ThreadPool(const std::string&, int) {}

		int ThreadPool::NumThreads() const {
			return 0;
		}

		void ThreadPool::Schedule(std::function<void()>) {}
		void ThreadPool::ParallelFor(int, std::function<void(int)>) {}

	}  // namespace concurrency
#endif
}  // namespace onnxruntime

#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/nn/pool.h"
#include "core/providers/cpu/nn/lrn.h"
#include "core/providers/cpu/nn/batch_norm.h"
#include "core/providers/cpu/nn/conv.h"

namespace onnxruntime {
template <>
Status Sum_6<float>::Compute(OpKernelContext*) const {
  __debugbreak();
  return Status::OK();
}

template <>
Status Pool<float, AveragePool>::Compute(OpKernelContext*) const {
  __debugbreak();
  return Status::OK();
}

template <>
Status Pool<float, MaxPool<1>>::Compute(OpKernelContext*) const {
  __debugbreak();
  return Status::OK();
}

template <>
Status Pool<float, MaxPool<8>>::Compute(OpKernelContext*) const {
  __debugbreak();
  return Status::OK();
}

template <>
Status LRN<float>::Compute(OpKernelContext*) const {
  __debugbreak();
  return Status::OK();
}

template <>
Status BatchNorm<float>::Compute(OpKernelContext*) const {
  __debugbreak();
  return Status::OK();
}

Status Conv<float>::Compute(OpKernelContext*) const {
  __debugbreak();
  return Status::OK();
}
}  // namespace onnxruntime
