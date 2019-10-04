// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

#include "core/framework/data_types.h"
#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"
#include "core/providers/bridge.h"

namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Mkldnn(int device_id);

static ProviderHost* g_host;

struct Provider_Mkldnn : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    return CreateExecutionProviderFactory_Mkldnn(device_id);
  }

  void SetProviderHost(ProviderHost& host) override {
    g_host = &host;
  }
} provider_;
}  // namespace onnxruntime

extern "C" {
__declspec(dllexport) onnxruntime::Provider* GetProvider() { return &onnxruntime::provider_; }
}

void* operator new(size_t n) { return onnxruntime::g_host->HeapAllocate(n); }
void operator delete(void* p) { return onnxruntime::g_host->HeapFree(p); }

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
#endif

template <>
MLDataType DataTypeImpl::GetTensorType<float>() { return nullptr; }

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
AttributeProto::AttributeProto() {}
AttributeProto::AttributeProto(const AttributeProto&) {}
AttributeProto::~AttributeProto() {}

void AttributeProto::CheckTypeAndMergeFrom(google::protobuf::MessageLite const&) {}
void AttributeProto::CopyFrom(AttributeProto const&) {}
void AttributeProto::Clear() {}
bool AttributeProto::IsInitialized() const { return false; }
unsigned __int64 AttributeProto::ByteSizeLong() const { return 0; }
bool AttributeProto::MergePartialFromCodedStream(google::protobuf::io::CodedInputStream*) { return false; }
void AttributeProto::SerializeWithCachedSizes(google::protobuf::io::CodedOutputStream*) const {}
bool AttributeProto_AttributeType_IsValid(int) { return false; }
std::string AttributeProto::GetTypeName() const { return ""; }

void TensorProto::CopyFrom(TensorProto const&) {}
}  // namespace onnx

google::protobuf::internal::LogMessage::LogMessage(google::protobuf::LogLevel, char const*, int) {}
google::protobuf::internal::LogMessage::~LogMessage() {}
google::protobuf::internal::LogMessage& google::protobuf::internal::LogMessage::operator<<(char const*) {
  return *this;
}

void google::protobuf::internal::LogFinisher::operator=(google::protobuf::internal::LogMessage&) {}

google::protobuf::MessageLite* google::protobuf::MessageLite::New(google::protobuf::Arena*) const {
  return nullptr;
}

std::string google::protobuf::MessageLite::InitializationErrorString() const {
  return "";
}

void google::protobuf::MessageLite::SerializeWithCachedSizes(google::protobuf::io::CodedOutputStream*) const {}
unsigned char* google::protobuf::MessageLite::SerializeWithCachedSizesToArray(unsigned char*) const {
  return nullptr;
}

unsigned char* google::protobuf::MessageLite::InternalSerializeWithCachedSizesToArray(bool, unsigned char*) const {
  return nullptr;
}
void google::protobuf::internal::RepeatedPtrFieldBase::Reserve(int) {}
template <>
onnx::AttributeProto* google::protobuf::Arena::CreateMaybeMessage<onnx::AttributeProto>(google::protobuf::Arena*) {
  return nullptr;
}
template <>
onnx::TensorProto* google::protobuf::Arena::CreateMaybeMessage<onnx::TensorProto>(google::protobuf::Arena*) {
  return nullptr;
}
google::protobuf::internal::ExplicitlyConstructed<std::string> google::protobuf::internal::fixed_address_empty_string;

namespace onnxruntime {

CPUIDInfo::CPUIDInfo() noexcept {}

#if 0
std::ostream& operator<<(std::ostream& stream, TensorShape const&) { return stream; }
#endif

namespace logging {
Logger* LoggingManager::s_default_logger_{};
const char* Category::onnxruntime{};
Capture::~Capture() {}

}  // namespace logging

namespace common {

Status::Status(StatusCategory, int, char const*) {}

Status::Status(StatusCategory category, int code, const std::string& msg) {
  __debugbreak();
}

std::string Status::ToString() const {
  return "";
}

const std::string& Status::ErrorMessage() const noexcept {
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

const OrtMemoryInfo& CPUAllocator::Info() const {
  return g_host->CPUAllocator_Info(this);
}

void* CPUAllocator::Alloc(uint64_t) { return nullptr; }
void CPUAllocator::Free(void*) {}

std::shared_ptr<IAllocator> CreateAllocator(DeviceAllocatorRegistrationInfo info, int device_id) {
  return g_host->CreateAllocator(std::move(info), device_id);
}

Status IExecutionProvider::Compile(const std::vector<Node*>&, std::string&) {
  return Status::OK();
}

Status IExecutionProvider::Compile(const std::vector<Node*>&, std::vector<NodeComputeInfo>&) { return Status::OK(); }

void IExecutionProvider::InsertAllocator(std::shared_ptr<IAllocator>) {}

Status IExecutionProvider::Sync() const { return Status::OK(); }
Status IExecutionProvider::OnRunStart() { return Status::OK(); }
Status IExecutionProvider::OnRunEnd() { return Status::OK(); }

std::shared_ptr<IAllocator> IExecutionProvider::GetAllocator(int, OrtMemType) const { return nullptr; }

std::shared_ptr<KernelRegistry> IExecutionProvider::GetKernelRegistry() const { return nullptr; }

std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider::GetCapability(const GraphViewer&, const std::vector<const KernelRegistry*>&) const { return {}; }

Status KernelRegistry::Register(KernelCreateInfo&&) { return Status::OK(); }

#if 0
const std::vector<MLDataType>& DataTypeImpl::AllTensorTypes() {
  __debugbreak();
  static std::vector<MLDataType> temp;
  return temp;
}
#endif

std::ostream& operator<<(std::ostream& out, const DataTypeImpl* data_type) { return out; }

TensorShape::TensorShape(__int64 const*, unsigned __int64) {}

int64_t TensorShape::Size() const { return 0; }

#if 0
int64_t TensorShape::SizeFromDimension(uint64_t) const { return 0; }
int64_t TensorShape::SizeToDimension(uint64_t) const { return 0; }

#endif

TensorShape TensorShape::Slice(unsigned __int64) const {
  return *(TensorShape*)nullptr;
}

#if 0
TensorShape TensorShape::Slice(uint64_t, uint64_t) const { return *(TensorShape*)nullptr; }
#endif

std::string TensorShape::ToString() const { return ""; }

#if 0
Status FeedsFetchesInfo::MapNamesToMLValueIdxs(const std::vector<std::string>&, const OrtValueNameIdxMap&, std::vector<int>&) { return Status::OK(); }

Status FeedsFetchesManager::Create(const std::vector<std::string>&, std::vector<std::string> const&, OrtValueNameIdxMap const&, std::unique_ptr<FeedsFetchesManager>&) { return Status::OK(); }

#endif

KernelDefBuilder& KernelDefBuilder::Provider(char const*) {
  return *this;
}

KernelDefBuilder& KernelDefBuilder::SetName(char const*) {
  return *this;
}

KernelDefBuilder& KernelDefBuilder::SetDomain(char const*) {
  return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(char const*, const DataTypeImpl*) {
  return *this;
}

#if 0
KernelDefBuilder& KernelDefBuilder::TypeConstraint(char const*, const std::vector<const DataTypeImpl*>&) { return *this; }

KernelDefBuilder& KernelDefBuilder::Alias(int, int) {
  return *this;
}

KernelDefBuilder& KernelDefBuilder::MayInplace(int, int) { return *this; }
#endif

const NodeAttributes& Node::GetAttributes() const noexcept { return *(NodeAttributes*)nullptr; }
NodeIndex Node::Index() const noexcept { return 0; }

const std::string& Node::OpType() const noexcept {
  static std::string dummy;
  return dummy;
}

const ONNX_NAMESPACE::OpSchema* Node::Op() const noexcept { return nullptr; }

const std::string&
NodeArg::Name() const noexcept {
  static std::string dummy;
  return dummy;
}

const ONNX_NAMESPACE::TensorShapeProto* NodeArg::Shape() const {
  return nullptr;
}

ONNX_NAMESPACE::DataType NodeArg::Type() const noexcept { return nullptr; }

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
  return 0;
}

const std::string& GraphViewer::Name() const noexcept {
  static std::string dummy;
  return dummy;
}

const Node* GraphViewer::GetNode(NodeIndex node_index) const { return nullptr; }

const InitializedTensorSet& GraphViewer::GetAllInitializedTensors() const noexcept { return *(InitializedTensorSet*)nullptr; }

#if 0

const GraphViewer* SessionState::GetGraphViewer() const {
  return nullptr;
}

SessionState const* SessionState::GetSubgraphSessionState(uint64_t, std::string const&) const { return nullptr; }
#endif
OpKernelInfo::OpKernelInfo(const OpKernelInfo& other) : OpKernelInfo(other.node_, other.kernel_def_, *other.execution_provider_, other.constant_initialized_tensors_,
                                                                     other.ort_value_name_idx_map_, other.funcs_mgr_, other.data_transfer_mgr_) {}
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
      proto_helper_context_(node) {}

const Node& OpKernelInfo::node() const noexcept { return *(Node*)nullptr; }

Tensor* OpKernelContext::Output(int, const TensorShape&) { return nullptr; }

#if 0

const DataTypeImpl* OpKernelContext::InputType(int) const { return nullptr; }

unsigned __int64 OpKernelContext::GetNodeIndex() const { return 0; }
#endif

const OrtValue* OpKernelContext::GetInputMLValue(int) const {
  return nullptr;
}

#if 0

OrtValue* OpKernelContext::GetOrCreateOutputMLValue(int) { return nullptr; }
#endif

const IExecutionProvider* OpKernelInfo::GetExecutionProvider() const noexcept { return nullptr; }

#if 0
const OrtValue* OpKernelContext::GetImplicitInputMLValue(int) const {
  return nullptr;
}

OrtValue* OpKernelContext::GetOutputMLValue(int) {
  return nullptr;
}

int OpKernelContext::NumVariadicInputs(uint64_t) const { return 0; }
#endif
Status OpKernelContext::GetTempSpaceAllocator(std::shared_ptr<IAllocator>*) const { return Status::OK(); }

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
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttr<float>(const std::string&, float*) const { return Status::OK(); }

template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttr<int64_t>(const std::string&, int64_t*) const { return Status::OK(); }

template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttr<std::string>(const std::string&, std::string*) const {
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
  return Status::OK();
}
template <>
template <>
Status OpNodeProtoHelper<ProtoHelperNodeContext>::GetAttrs<int64_t>(const std::string&, std::vector<int64_t>&) const {
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
    : ort_value_{&ort_value},
      position_{position},
      increment_by_{direction == Direction::kForward ? 1 : -1},
      position_materialized_{-1} {
}

template <>
OrtValueTensorSlicer<const OrtValue>::Iterator::Iterator(const OrtValue& ort_value, size_t, size_t, int64_t position, OrtValueTensorSlicer<const OrtValue>::Iterator::Direction direction)
    : ort_value_{&ort_value},
      position_{position},
      increment_by_{direction == Direction::kForward ? 1 : -1},
      position_materialized_{-1} {
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
const KernelDef& OpKernelInfo::GetKernelDef() const { return *(KernelDef*)nullptr; }

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
Status Sum_6<float>::Compute(OpKernelContext*) const { return Status::OK(); }

template <>
Status Pool<float, AveragePool>::Compute(OpKernelContext*) const {
  return Status::OK();
}

template <>
Status Pool<float, MaxPool<1>>::Compute(OpKernelContext*) const {
  return Status::OK();
}

template <>
Status Pool<float, MaxPool<8>>::Compute(OpKernelContext*) const {
  return Status::OK();
}

template <>
Status LRN<float>::Compute(OpKernelContext*) const {
  return Status::OK();
}

template <>
Status BatchNorm<float>::Compute(OpKernelContext*) const {
  return Status::OK();
}

Status Conv<float>::Compute(OpKernelContext*) const {
  return Status::OK();
}
}  // namespace onnxruntime

#include "core/framework/allocator.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/providers/mkldnn/subgraph/mkldnn_func_kernel.h"
#include "mkldnn_execution_provider.h"
#include "mkldnn_fwd.h"

namespace onnxruntime {
constexpr const char* MKLDNN = "MklDnn";
constexpr const char* MKLDNN_CPU = "MklDnnCpu";

MKLDNNExecutionProvider::MKLDNNExecutionProvider(const MKLDNNExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kMklDnnExecutionProvider} {
  DeviceAllocatorRegistrationInfo default_memory_info({OrtMemTypeDefault,
                                                       [](int) { return std::make_unique<CPUAllocator>(std::make_unique<OrtMemoryInfo>(MKLDNN, OrtAllocatorType::OrtDeviceAllocator)); }, std::numeric_limits<size_t>::max()});

  DeviceAllocatorRegistrationInfo cpu_memory_info({OrtMemTypeCPUOutput,
                                                   [](int) { return std::make_unique<CPUAllocator>(std::make_unique<OrtMemoryInfo>(MKLDNN_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput)); }, std::numeric_limits<size_t>::max()});

  if (info.create_arena) {
    InsertAllocator(CreateAllocator(default_memory_info));

    InsertAllocator(CreateAllocator(cpu_memory_info));
  } else {
    InsertAllocator(std::shared_ptr<IArenaAllocator>(
        std::make_unique<DummyArena>(default_memory_info.factory(0))));

    InsertAllocator(std::shared_ptr<IArenaAllocator>(
        std::make_unique<DummyArena>(cpu_memory_info.factory(0))));
  }
}  // namespace onnxruntime

MKLDNNExecutionProvider::~MKLDNNExecutionProvider() {
}

namespace mkl_dnn {
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 6, Relu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 6, Sum);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, 8, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 8, 8, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, float, LRN);

void RegisterMKLDNNKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 6, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 6, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, 8, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 8, 8, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, float, LRN)>,
  };

  for (auto& function_table_entry : function_table) {
    kernel_registry.Register(function_table_entry());
  }
}

std::shared_ptr<KernelRegistry> GetMklDnnKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterMKLDNNKernels(*kernel_registry);
  return kernel_registry;
}
}  // namespace mkl_dnn

std::shared_ptr<KernelRegistry> MKLDNNExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::mkl_dnn::GetMklDnnKernelRegistry();
  return kernel_registry;
}

bool MKLDNNExecutionProvider::UseSubgraph(const onnxruntime::GraphViewer& graph_viewer) const {
  // switch between mkldnn-vanilla and mkldnn-subgraph implementation using
  // MKLDNN_SUBGRAPH environment variable
  bool use_subgraph = true;

  bool FP16_graph = false;
  bool mkldnn_nodes_in_the_graph = false;

  if (graph_viewer.MaxNodeIndex() > 0) {
    int index = 0;
    auto node = graph_viewer.GetNode(index);
    while (node == NULL) {
      index++;
      node = graph_viewer.GetNode(index);
    }
    if (!node->InputDefs().empty() && node->InputDefs()[0]->Type() != nullptr)
      FP16_graph = node->InputDefs()[0]->Type()->find("16") != std::string::npos;
  }

  for (auto node_index = 0; node_index < graph_viewer.MaxNodeIndex(); node_index++) {
    auto node = graph_viewer.GetNode(node_index);
    if (node == nullptr) {
      node_index++;
      continue;
    }
    auto op_it = mkldnn_ops_.find(node->OpType());
    if (op_it != mkldnn_ops_.end()) {
      mkldnn_nodes_in_the_graph = true;
      break;
    }
  }

  if (FP16_graph || !mkldnn_nodes_in_the_graph) {
    // FP16 not supported yet.
    use_subgraph = false;
  } else {
    const char* env = getenv("ORT_MKLDNN_SUBGRAPH");
    if (env != nullptr) {
      if (atoi(env) == 0) {
        use_subgraph = false;
      }
    }
  }
  return use_subgraph;
}

void MKLDNNExecutionProvider::CreateOrUpdateMklDnnNode(const Node* node,
                                                       std::shared_ptr<mkl_dnn::Subgraph>& subgraph_ptr,
                                                       mkl_dnn::Subgraph::SubgraphVariables& sub_var,
                                                       bool fused,
                                                       std::map<std::string, size_t>& output_to_source_node_map,
                                                       NodeAttributes& subgraph_attributes) const {
  const auto& node_inputs = node->InputDefs();
  sub_var.outputs.push_back(node->OutputDefs()[0]->Name());

  if (!fused) {
    mkl_dnn::MklDnnNode mkldnn_node;
    mkldnn_node.name = node->OpType();
    mkldnn_node.num_inputs = static_cast<int>(node->InputDefs().size());
    mkldnn_node.input_start_index = static_cast<int>(sub_var.inputs.size()) - 1;
    mkldnn_node.node_index = static_cast<int>(subgraph_ptr->mkldnn_nodes.size()) + 1;
    const auto& node_outputs = node->OutputDefs();
    mkldnn_node.output_name = node_outputs[0]->Name();
    if (node->OpType() == "Conv") {
      mkldnn_node.weight_name = node->InputDefs()[1]->Name();
    }
    for (size_t i = 0; i < node_inputs.size(); i++) {
      auto iter = output_to_source_node_map.find(node_inputs[i]->Name());
      if (iter != output_to_source_node_map.end())
        mkldnn_node.parent_nodes.push_back(iter->second);
    }
    subgraph_ptr->mkldnn_nodes.push_back(mkldnn_node);
    output_to_source_node_map.insert(std::make_pair(node_outputs[0]->Name(), subgraph_ptr->mkldnn_nodes.size() - 1));
  } else {
    const auto& node_outputs = node->OutputDefs();
    output_to_source_node_map.erase(subgraph_ptr->mkldnn_nodes.back().output_name);
    subgraph_ptr->mkldnn_nodes.back().output_name = node_outputs[0]->Name();
    output_to_source_node_map.insert(std::make_pair(node_outputs[0]->Name(), subgraph_ptr->mkldnn_nodes.size() - 1));
  }

  // Add inputs which are not in the outputs vector.
  for (size_t i = 0; i < node_inputs.size(); i++) {
    auto itr = std::find(sub_var.outputs.begin(), sub_var.outputs.end(), node_inputs[i]->Name());
    if (itr == sub_var.outputs.end()) {
      sub_var.inputs.push_back(node_inputs[i]->Name());
    } else {
      // Vector of node outputs, which is input to other node
      // if node output is not input to any other node, then it's the end node
      // which we will find later
      sub_var.outputs_as_input_other_node.push_back(node_inputs[i]->Name());
    }
  }

  NodeAttributes attributes = node->GetAttributes();
  if (attributes.size() > 0) {
    size_t index = subgraph_ptr->mkldnn_nodes.size();

    for (auto att_it = attributes.begin(); att_it != attributes.end(); ++att_it) {
      std::string key = node->OpType() + "-" + std::to_string(index) + "-" + att_it->first;
      std::pair<std::string, ONNX_NAMESPACE::AttributeProto> att(key, att_it->second);
      subgraph_attributes[key] = att_it->second;
    }
  }
}

std::vector<std::unique_ptr<ComputeCapability>> MKLDNNExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const std::vector<const KernelRegistry*>& kernel_registries) const {
  ORT_UNUSED_PARAMETER(kernel_registries);

  // temporary switch to toggle between mkldnn-vanilla and mkldnn-subgraph implementation using
  // ORT_MKLDNN_SUBGRAPH environment variable
  if (UseSubgraph(graph_viewer) == false) {
    return IExecutionProvider::GetCapability(graph_viewer, kernel_registries);
  }

  LOGS_DEFAULT(INFO) << "Using MKL-DNN Subgraph";
  // use sub-graph implementation
  std::vector<std::unique_ptr<ComputeCapability>> result;
  mkl_dnn::Subgraph::SubgraphVariables sub_var;
  std::shared_ptr<mkl_dnn::Subgraph> subgraph_ptr;

  // We need graph name make PrimitivePool keys unique.
  // There are several identical graphs in Model zoo and only differ in
  // few attribute values. GetGraphName return graph-name + first-node-output name
  std::string graph_name = GetGraphName(graph_viewer);
  subgraph_ptr.reset(new mkl_dnn::Subgraph(graph_name));

  // output name to node index map. Using it to find sub-graph end nodes
  // if output of a node is not an input to any node in a sub-graph is end node
  std::map<std::string, size_t> output_to_source_node_map;
  NodeAttributes subgraph_attributes;
  int node_index = 0;

  while (node_index < graph_viewer.MaxNodeIndex()) {
    auto node = graph_viewer.GetNode(node_index);
    if (node == nullptr) {
      node_index++;
      continue;
    }

    if (IsDimensionSupported(node) == false) {
      node_index++;
      if (subgraph_ptr->mkldnn_nodes.size() > 0) {
        CreateMetaDef(graph_viewer, subgraph_attributes, subgraph_ptr, sub_var, result);
        subgraph_ptr.reset(new mkl_dnn::Subgraph(graph_name));
        subgraph_attributes.clear();
        output_to_source_node_map.clear();
      }
      continue;
    }

    auto op_it = mkldnn_ops_.find(node->OpType());
    if (op_it != mkldnn_ops_.end()) {
      sub_var.subgraph_node_indexes.push_back(node->Index());

      // can we fuse (at mkldnn level) nodes?
      bool fused = false;
      if (sub_var.subgraph_node_indexes.size() > 1 && node->OpType() == "Relu") {
        if (subgraph_ptr->mkldnn_nodes.back().name == "BatchNormalization" || subgraph_ptr->mkldnn_nodes.back().name == "Conv") {
          subgraph_ptr->mkldnn_nodes.back().name += "-Relu";
          fused = true;
        }
      }

      // Create MklDnn node:
      //   Update inputs, outputs and parent nodes
      //   Collect attributes and modify the key to make it unique
      CreateOrUpdateMklDnnNode(node, subgraph_ptr, sub_var, fused, output_to_source_node_map, subgraph_attributes);

      auto temp_index = node_index + 1;
      if (temp_index < graph_viewer.MaxNodeIndex()) {
        if (!sub_var.subgraph_node_indexes.empty()) {
          // if next node is mkldnn node and if it's input is not output of current node
          //   if next node input is output of any of the nodes in sub-graph continue
          // else
          //   break and create sub-graph
          auto next_node = graph_viewer.GetNode(temp_index);
          while (next_node == nullptr) {
            temp_index++;
            next_node = graph_viewer.GetNode(temp_index);
          }
          auto sub_it = mkldnn_ops_.find(next_node->OpType());
          if (sub_it != mkldnn_ops_.end()) {
            const auto& next_node_inputs = next_node->InputDefs();
            bool input_from_subgraph = true;
            size_t inputs_count = 1;
            if (next_node->OpType() == "Sum")
              inputs_count = next_node_inputs.size();
            for (size_t i = 0; i < inputs_count; i++) {
              auto in = next_node_inputs[i];
              auto itr = std::find(sub_var.outputs.begin(), sub_var.outputs.end(), in->Name());
              if (itr == sub_var.outputs.end()) {
                input_from_subgraph = false;
              }
            }
            if (input_from_subgraph == false) {
              CreateMetaDef(graph_viewer, subgraph_attributes, subgraph_ptr, sub_var, result);
              subgraph_attributes.clear();
              subgraph_ptr.reset(new mkl_dnn::Subgraph(graph_name));
              output_to_source_node_map.clear();
            }
          }
        }
        if (!sub_var.subgraph_node_indexes.empty()) {
          if (node->GetOutputEdgesCount() > 1) {
            // If current node has branches
            //    iterate and see if all nodes are mkldnn ops OR
            //      it ends in node with same number of input edges (mkldnn node or cpu node)
            //      create sub-graph
            bool create_subgraph = false;
            bool break_loop = false;
            while (!break_loop) {
              if (temp_index > graph_viewer.MaxNodeIndex())
                break_loop = true;

              auto next_node = graph_viewer.GetNode(temp_index);
              while (next_node == nullptr) {
                temp_index++;
                next_node = graph_viewer.GetNode(temp_index);
              }
              if (next_node->GetInputEdgesCount() == node->GetOutputEdgesCount()) {
                // if all nodes in the branch loop are mkldnn nodes
                // then continue with adding nodes to sub-graph
                break_loop = true;
              }
              // inner nodes. if inner nodes are not  mkldnn nodes
              // create subgraph (inception v2)
              auto sub_it = mkldnn_ops_.find(next_node->OpType());
              if (sub_it == mkldnn_ops_.end()) {
                // break and create a sub-graph
                break_loop = true;
                create_subgraph = true;
              }
              temp_index++;
            }
            if (create_subgraph) {
              CreateMetaDef(graph_viewer, subgraph_attributes, subgraph_ptr, sub_var, result);
              subgraph_ptr.reset(new mkl_dnn::Subgraph(graph_name));
              subgraph_attributes.clear();
              output_to_source_node_map.clear();
            }
          }
        }
      }
    } else {
      if (!sub_var.subgraph_node_indexes.empty()) {
        CreateMetaDef(graph_viewer, subgraph_attributes, subgraph_ptr, sub_var, result);
        subgraph_ptr.reset(new mkl_dnn::Subgraph(graph_name));
        subgraph_attributes.clear();
        output_to_source_node_map.clear();
      }
    }
    node_index++;
  }  // graph_viewer node iterator ends
  if (!sub_var.subgraph_node_indexes.empty()) {
    CreateMetaDef(graph_viewer, subgraph_attributes, subgraph_ptr, sub_var, result);
    subgraph_ptr.reset(new mkl_dnn::Subgraph(graph_name));
    subgraph_attributes.clear();
    output_to_source_node_map.clear();
  }
  return result;
}

void MKLDNNExecutionProvider::CreateMetaDef(const onnxruntime::GraphViewer& graph_viewer,
                                            const NodeAttributes& subgraph_attributes,
                                            std::shared_ptr<mkl_dnn::Subgraph>& subgraph_ptr,
                                            mkl_dnn::Subgraph::SubgraphVariables& sub_var,
                                            std::vector<std::unique_ptr<ComputeCapability>>& result) const {
  std::string graph_fused_nodes;
  std::string node_list;
  std::string subgraph_id = std::to_string(subgraph_index_);
  subgraph_index_++;

  // This is a list of initializers that subgraph considers as constants.
  // Example weights, reshape shape etc.
  std::unordered_set<std::string> input_initializers;

  // Create ng_required_initializers attribute of NGraphCustomOp
  ONNX_NAMESPACE::AttributeProto initializers;
  initializers.set_name("initializers");
  initializers.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSORS);

  for (const auto& init : sub_var.inputs) {
    if (graph_viewer.GetAllInitializedTensors().count(init)) {
      auto tensor = initializers.add_tensors();
      *tensor = *(graph_viewer.GetAllInitializedTensors().at(init));
    }
  }

  auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->attributes["initializers"] = initializers;
  meta_def->name = "MkldnnCustomOp" + std::to_string(subgraph_index_);
  meta_def->domain = kMSDomain;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->inputs = sub_var.inputs;
  meta_def->attributes.insert(subgraph_attributes.begin(), subgraph_attributes.end());

  // Find the end nodes
  for (auto& mklnode : subgraph_ptr->mkldnn_nodes) {
    auto itr = std::find(sub_var.outputs_as_input_other_node.begin(),
                         sub_var.outputs_as_input_other_node.end(), mklnode.output_name);
    if (itr == sub_var.outputs_as_input_other_node.end()) {
      meta_def->outputs.push_back(mklnode.output_name);
      mklnode.output_index = static_cast<int>(meta_def->outputs.size()) - 1;
    }
  }

  ONNX_NAMESPACE::AttributeProto ap;
  ap.set_s(subgraph_id);
  ap.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
  meta_def->attributes["subgraph_id"] = ap;
  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
  sub_graph->nodes = sub_var.subgraph_node_indexes;
  sub_graph->SetMetaDef(meta_def);
  result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  mkl_subgraphs_.insert(std::make_pair(subgraph_id, subgraph_ptr));

  // Reset subgraph and meta_Def
  sub_var.Reset();
}

Status MKLDNNExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                        std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto* fused_node : fused_nodes) {
    auto attributes = fused_node->GetAttributes();
    NodeComputeInfo compute_info;

    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      auto* p = new onnxruntime::mkl_dnn::MkldnnFuncKernel<float>(context, attributes, this);
      *state = p;
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete static_cast<onnxruntime::mkl_dnn::MkldnnFuncKernel<float>*>(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      onnxruntime::mkl_dnn::MkldnnFuncKernel<float>* custom_op = reinterpret_cast<mkl_dnn::MkldnnFuncKernel<float>*>(state);
      return custom_op->Compute(api, context);
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}
}  // namespace onnxruntime
