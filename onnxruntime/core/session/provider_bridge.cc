#include "core/framework/data_types.h"
#include "core/framework/allocatormgr.h"
#include "core/providers/bridge.h"
#include "core/providers/mkldnn/mkldnn_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "core/platform/env.h"

namespace onnxruntime {

struct ProviderHostImpl : ProviderHost {
  MLDataType DataTypeImpl_GetType_Tensor() override { return DataTypeImpl::GetType<Tensor>(); }
  MLDataType DataTypeImpl_GetType_float() override { return DataTypeImpl::GetType<float>(); }

  void* HeapAllocate(size_t size) override { return new uint8_t[size]; }
  void HeapFree(void* p) override { delete p; }

  std::shared_ptr<IAllocator> CreateAllocator(DeviceAllocatorRegistrationInfo&& info, int device_id) override {
    return onnxruntime::CreateAllocator(std::move(info), device_id);
  }

  virtual const OrtMemoryInfo& CPUAllocator_Info(const CPUAllocator* _this) override {
    return _this->Info();
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
