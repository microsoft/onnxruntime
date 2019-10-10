#include "core/framework/data_types.h"
#include "core/framework/allocatormgr.h"
#include "core/providers/mkldnn/mkldnn_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "core/platform/env.h"
#include "core/framework/execution_provider.h"
#include "core/framework/compute_capability.h"
#include "core/providers/shared_library/bridge.h"
#include "core/common/logging/logging.h"
#include "core/common/cpuid_info.h"

using namespace google::protobuf::internal;

template <typename PtrType, PtrType Value, typename TagType>
struct private_access_helper {
  friend PtrType private_cast(TagType) {
    return Value;
  }
};

// actually defines the class and therefore defines the private_cast() function
struct private_cast_google_protobuf_internal_RepeatedPtrFieldBase_Reserve {};
template struct private_access_helper<void (RepeatedPtrFieldBase::*)(int), &RepeatedPtrFieldBase::Reserve, private_cast_google_protobuf_internal_RepeatedPtrFieldBase_Reserve>;
struct private_cast_google_protobuf_Arena_CreateMaybeMessage_onnx_TensorProto {};
template struct private_access_helper<onnx::TensorProto* (*)(google::protobuf::Arena*), &google::protobuf::Arena::CreateMaybeMessage<onnx::TensorProto>, private_cast_google_protobuf_Arena_CreateMaybeMessage_onnx_TensorProto>;

namespace onnxruntime {

struct ProviderHostImpl : ProviderHost {
  ProviderHostImpl::ProviderHostImpl() {
    google_protobuf_internal_RepeatedPtrFieldBase_Reserve = private_cast(private_cast_google_protobuf_internal_RepeatedPtrFieldBase_Reserve{});
    google_protobuf_Arena_CreateMaybeMessage_onnx_TensorProto = private_cast(private_cast_google_protobuf_Arena_CreateMaybeMessage_onnx_TensorProto{});

    DataTypeImpl_GetType_Tensor = &DataTypeImpl::GetType<Tensor>;
    DataTypeImpl_GetType_float = &DataTypeImpl::GetType<float>;
    DataTypeImpl_GetTensorType_float = &DataTypeImpl::GetTensorType<float>;

    onnx_AttributeProto_CopyFrom = &onnx::AttributeProto::CopyFrom;
    onnx_TensorProto_CopyFrom = &onnx::TensorProto::CopyFrom;
  }

  const ::std::string& google_protobuf_internal_GetEmptyStringAlreadyInited() override {
    return google::protobuf::internal::GetEmptyStringAlreadyInited();
  }

  logging::Logger* LoggingManager_GetDefaultLogger() override { return const_cast<logging::Logger*>(&logging::LoggingManager::DefaultLogger()); }

  void* HeapAllocate(size_t size) override { return new uint8_t[size]; }
  void HeapFree(void* p) override { delete p; }

  void onnx_AttributeProto_constructor(void* _this) override {
    new (_this) onnx::AttributeProto();
  }

  void onnx_AttributeProto_copy_constructor(void* _this, void* copy) override {
    new (_this) onnx::AttributeProto(*reinterpret_cast<const onnx::AttributeProto*>(copy));
  }

  void onnx_AttributeProto_destructor(void* _this) override {
    reinterpret_cast<onnx::AttributeProto*>(_this)->AttributeProto::~AttributeProto();
  }

  bool onnx_AttributeProto_AttributeType_IsValid(int p1) override {
    return onnx::AttributeProto_AttributeType_IsValid(p1);
  }

  std::shared_ptr<IAllocator> CreateAllocator(DeviceAllocatorRegistrationInfo&& info, int device_id) override {
    return onnxruntime::CreateAllocator(std::move(info), device_id);
  }

  const CPUIDInfo& CPUIDInfo_GetCPUIDInfo() override {
    return CPUIDInfo::GetCPUIDInfo();
  }

  void* CPUAllocator_Alloc(CPUAllocator* _this, uint64_t p1) override {
    return _this->CPUAllocator::Alloc(p1);
  }

  void CPUAllocator_Free(CPUAllocator* _this, void* p1) override {
    return _this->CPUAllocator::Free(p1);
  }

  const OrtMemoryInfo& CPUAllocator_Info(const CPUAllocator* _this) override {
    return _this->CPUAllocator::Info();
  }

  std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider_GetCapability(const IExecutionProvider* _this, const GraphViewer& p1, const std::vector<const KernelRegistry*>& p2) override {
    return _this->IExecutionProvider::GetCapability(p1, p2);
  }

  std::shared_ptr<IAllocator> IExecutionProvider_GetAllocator(const IExecutionProvider* _this, int p1, OrtMemType p2) override {
    return _this->IExecutionProvider::GetAllocator(p1, p2);
  }

  void IExecutionProvider_InsertAllocator(IExecutionProvider* _this, std::shared_ptr<IAllocator> p1) override {
    _this->IExecutionProvider::InsertAllocator(std::move(p1));
  }

  Status IExecutionProvider_OnRunEnd(IExecutionProvider* _this) override {
    return _this->IExecutionProvider::OnRunEnd();
  }

  Status IExecutionProvider_OnRunStart(IExecutionProvider* _this) override {
    return _this->IExecutionProvider::OnRunStart();
  }

  const InitializedTensorSet& GraphViewer_GetAllInitializedTensors(const GraphViewer* _this) override {
    return _this->GraphViewer::GetAllInitializedTensors();
  }

  const Node* GraphViewer_GetNode(const GraphViewer* _this, NodeIndex p1) override {
    return _this->GraphViewer::GetNode(p1);
  }

  int GraphViewer_MaxNodeIndex(const GraphViewer* _this) override {
    return _this->GraphViewer::MaxNodeIndex();
  }

  const std::string& GraphViewer_Name(const GraphViewer* _this) override {
    return _this->GraphViewer::Name();
  }

  KernelDefBuilder& KernelDefBuilder_Provider(KernelDefBuilder* _this, char const* p1) override {
    return _this->KernelDefBuilder::Provider(p1);
  }

  KernelDefBuilder& KernelDefBuilder_SetName(KernelDefBuilder* _this, char const* p1) override {
    return _this->KernelDefBuilder::SetName(p1);
  }

  KernelDefBuilder& KernelDefBuilder_SetDomain(KernelDefBuilder* _this, char const* p1) override {
    return _this->KernelDefBuilder::SetDomain(p1);
  }

  KernelDefBuilder& KernelDefBuilder_TypeConstraint(KernelDefBuilder* _this, char const* p1, const DataTypeImpl* p2) override {
    return _this->KernelDefBuilder::TypeConstraint(p1, p2);
  }

  Status KernelRegistry_Register(KernelRegistry* _this, KernelCreateInfo&& p1) override {
    return _this->KernelRegistry::Register(std::move(p1));
  }

  NodeIndex Node_Index(const Node* _this) override {
    return _this->Node::Index();
  }

  const NodeAttributes& Node_GetAttributes(const Node* _this) override {
    return _this->Node::GetAttributes();
  }

  const ONNX_NAMESPACE::OpSchema* Node_Op(const Node* _this) override {
    return _this->Node::Op();
  }

  const std::string& Node_OpType(const Node* _this) override {
    return _this->Node::OpType();
  }

  const std::string& NodeArg_Name(const NodeArg* _this) override {
    return _this->NodeArg::Name();
  }

  const ONNX_NAMESPACE::TensorShapeProto* NodeArg_Shape(const NodeArg* _this) override {
    return _this->NodeArg::Shape();
  }

  ONNX_NAMESPACE::DataType NodeArg_Type(const NodeArg* _this) override {
    return _this->NodeArg::Type();
  }

  void onnxruntime_TensorShape_constructor(void* _this, __int64 const* p1, unsigned __int64 p2) override {
    new (_this) TensorShape(p1, p2);
  }

  int64_t TensorShape_Size(const TensorShape* _this) override {
    return _this->TensorShape::Size();
  }

  TensorShape TensorShape_Slice(const TensorShape* _this, unsigned __int64 p1) override {
    return _this->TensorShape::Slice(p1);
  }

#if 0

  MLDataType GetType(DataTypes::Type type) override {
    return types_[type];
  }

  MLDataType GetTensorType(TensorDataTypes::Type type) override {
    return tensorTypes_[type];
  }

  __int64 TensorShape_Size(const TensorShape* pThis) override {
    return pThis->Size();
  }

  const MLValue* OpKernelContext_GetInputMLValue(const OpKernelContext* pThis, int index) override {
    return pThis->GetInputMLValue(index);
  }

  Tensor* OpKernelContext_Output(OpKernelContext* pThis, int index, const TensorShape& shape) override {
    return pThis->Output(index, shape);
  }

  void AttributeProto_Destructor(onnx::AttributeProto* pThis) override {
    pThis->~AttributeProto();
  }

  ::onnx::OpSchema& OpSchema_SinceVersion(::onnx::OpSchema* pThis, ::onnx::OperatorSetVersion n) override {
    return pThis->SinceVersion(n);
  }

  ::onnx::OpSchema& OpSchema_Input(::onnx::OpSchema* pThis,
                                   int n,
                                   const char* name,
                                   const char* description,
                                   const char* type_str,
                                   ::onnx::OpSchema::FormalParameterOption param_option) override {
    return pThis->Input(n, name, description, type_str, param_option);
  }

  ::onnx::OpSchema& OpSchema_Output(::onnx::OpSchema* pThis,
                                    int n,
                                    const char* name,
                                    const char* description,
                                    const char* type_str,
                                    ::onnx::OpSchema::FormalParameterOption param_option) override {
    return pThis->Output(n, name, description, type_str, param_option);
  }

  ::onnx::OpSchema& OpSchema_TypeConstraint(::onnx::OpSchema* pThis,
                                            std::string type_str,
                                            std::vector<std::string> constraints,
                                            std::string description) override {
    return pThis->TypeConstraint(type_str, constraints, description);
  }

  void AttributeProto_AttributeProto(::onnx::AttributeProto* pThis, const ::onnx::AttributeProto& copy) override {
    new (pThis)::onnx::AttributeProto(copy);
  }

  ::google::protobuf::uint8* AttributeProto_InternalSerializeWithCachedSizesToArray(const ::onnx::AttributeProto* pThis,
                                                                                    bool deterministic, ::google::protobuf::uint8* target) override {
    return pThis->InternalSerializeWithCachedSizesToArray(deterministic, target);
  }

  ::google::protobuf::Metadata AttributeProto_GetMetadata(const ::onnx::AttributeProto* pThis) override {
    return pThis->GetMetadata();
  }

  bool AttributeProto_MergePartialFromCodedStream(::onnx::AttributeProto* pThis, ::google::protobuf::io::CodedInputStream* input) override {
    return pThis->MergePartialFromCodedStream(input);
  }

  void AttributeProto_SerializeWithCachedSizes(const ::onnx::AttributeProto* pThis, ::google::protobuf::io::CodedOutputStream* output) override {
    pThis->SerializeWithCachedSizes(output);
  }

  size_t AttributeProto_ByteSizeLong(const ::onnx::AttributeProto* pThis) override {
    return pThis->ByteSizeLong();
  }

  bool AttributeProto_IsInitialized(const ::onnx::AttributeProto* pThis) override {
    return pThis->IsInitialized();
  }

  void AttributeProto_Clear(::onnx::AttributeProto* pThis) override {
    pThis->Clear();
  }

  void AttributeProto_CopyFrom(::onnx::AttributeProto* pThis, const ::google::protobuf::Message& message) override {
    pThis->CopyFrom(message);
  }

  void AttributeProto_MergeFrom(::onnx::AttributeProto* pThis, const ::google::protobuf::Message& message) override {
    pThis->MergeFrom(message);
  }

#endif

} provider_host_;

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Mkldnn(int device_id) {
  void* handle;
  Env::Default().LoadDynamicLibrary("C:\\code\\github\\onnxrt\\build\\windows\\debug\\debug\\onnxruntime_providers_mkldnn.dll", &handle);
  if (!handle)
    return nullptr;

  Provider* (*PGetProvider)();

  Env::Default().GetSymbolFromLibrary(handle, "GetProvider", (void**)&PGetProvider);
  //return std::make_shared<onnxruntime::MkldnnProviderFactory>(device_id);
  //TODO: This is apparently a bug. The consructor parameter is create-arena-flag, not the device-id

  Provider* provider = PGetProvider();
  provider->SetProviderHost(provider_host_);

  return provider->CreateExecutionProviderFactory(device_id);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Mkldnn, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Mkldnn(use_arena));
  return nullptr;
}
