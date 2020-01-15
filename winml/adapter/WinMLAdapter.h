// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

namespace Windows::AI::MachineLearning::Adapter {
TRACELOGGING_DECLARE_PROVIDER(winml_trace_logging_provider);

MIDL_INTERFACE("eaae30b5-7381-432d-9730-322136b02371") IModelInfo : IUnknown{
    // model metadata
    virtual const char* STDMETHODCALLTYPE author() = 0;
    virtual const char* STDMETHODCALLTYPE name() = 0;
    virtual const char* STDMETHODCALLTYPE domain() = 0;
    virtual const char* STDMETHODCALLTYPE description() = 0;
    virtual int64_t STDMETHODCALLTYPE version() = 0;    
    virtual HRESULT STDMETHODCALLTYPE GetModelMetadata(ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING> ** metadata) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetInputFeatures(ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor> * *features) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetOutputFeatures(ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor> * *features) = 0;
};

MIDL_INTERFACE("a848faf6-5a2e-4a7f-b622-cc036f71e28a") IModelProto : IUnknown{
    // this returns a weak ref
    virtual onnx::ModelProto* STDMETHODCALLTYPE get() = 0;
    // this returns the ownership without touching the reference and forgets about the object
    virtual onnx::ModelProto* STDMETHODCALLTYPE detach() = 0;
};

MIDL_INTERFACE("6ec766ef-6365-42bf-b64f-ae85c015adb8") IInferenceSession : IUnknown {
    virtual onnxruntime::InferenceSession* STDMETHODCALLTYPE get() = 0;
    // the below returns a weak ref , DO NOT RELEASE IT
    virtual HRESULT STDMETHODCALLTYPE GetOrtSession(OrtSession ** out) = 0;
    virtual void STDMETHODCALLTYPE RegisterGraphTransformers() = 0;
    virtual HRESULT STDMETHODCALLTYPE RegisterCustomRegistry(IMLOperatorRegistry * registry) = 0;
    virtual HRESULT STDMETHODCALLTYPE LoadModel(IModelProto* model_proto) = 0;
    virtual HRESULT STDMETHODCALLTYPE StartProfiling() = 0;
    virtual HRESULT STDMETHODCALLTYPE EndProfiling() = 0;
    virtual void STDMETHODCALLTYPE FlushContext(onnxruntime::IExecutionProvider * dml_provider) = 0;
    virtual void STDMETHODCALLTYPE TrimUploadHeap(onnxruntime::IExecutionProvider* dml_provider) = 0;
    virtual void STDMETHODCALLTYPE ReleaseCompletedReferences(onnxruntime::IExecutionProvider* dml_provider) = 0;
    virtual HRESULT STDMETHODCALLTYPE CopyOneInputAcrossDevices(const char* input_name,
                                                                const OrtValue* orig_mlvalue, OrtValue** new_mlvalue) = 0;
};

// The IOrtSessionBuilder offers an abstraction over the creation of
// InferenceSession, that enables the creation of the session based on a device (CPU/DML).
MIDL_INTERFACE("2746f03a-7e08-4564-b5d0-c670fef116ee") IOrtSessionBuilder : IUnknown {

  virtual HRESULT STDMETHODCALLTYPE CreateSessionOptions(
      OrtSessionOptions ** options) = 0;

  virtual HRESULT STDMETHODCALLTYPE CreateSession(
      OrtSessionOptions * options,
      IInferenceSession** session,
      onnxruntime::IExecutionProvider** provider) = 0;

  virtual HRESULT STDMETHODCALLTYPE Initialize(
      IInferenceSession* session,
      onnxruntime::IExecutionProvider* provider) = 0;
};


MIDL_INTERFACE("b19385e7-d9af-441a-ba7f-3993c7b1c9db") IWinMLAdapter : IUnknown {

    virtual void STDMETHODCALLTYPE EnableDebugOutput() = 0;

    virtual HRESULT STDMETHODCALLTYPE EnsureModelDeviceCompatibility(
        winml::LearningModel const& model,
        IModelProto* p_model_proto,
        bool is_float16_supported) = 0;

    // factory method for creating an ortsessionbuilder from a device
    virtual HRESULT STDMETHODCALLTYPE CreateOrtSessionBuilder(
            ID3D12Device* device,
            ID3D12CommandQueue* queue,
            IOrtSessionBuilder** session_builder) = 0;

    // factory methods for creating model protos
    virtual HRESULT STDMETHODCALLTYPE CreateModelProto(const char* path, IModelProto** model_proto) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateModelProto(ABI::Windows::Storage::Streams::IRandomAccessStreamReference* stream_reference, IModelProto** model_proto) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateModelProto(IModelProto * model_proto_in, IModelProto** model_proto) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateModelInfo(IModelProto * model_proto, IModelInfo ** model_info) = 0;

    // Data types

    // custom ops
    virtual HRESULT STDMETHODCALLTYPE GetCustomRegistry(IMLOperatorRegistry** registry) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetOperatorRegistry(ILearningModelOperatorProviderNative * operator_provider_native, IMLOperatorRegistry * *registry) = 0;

    // dml ep hooks
    virtual void* STDMETHODCALLTYPE CreateGPUAllocationFromD3DResource(ID3D12Resource* pResource) = 0;
    virtual void STDMETHODCALLTYPE FreeGPUAllocation(void* ptr) = 0;
    virtual HRESULT STDMETHODCALLTYPE CopyTensor(onnxruntime::IExecutionProvider* provider, OrtValue* src, OrtValue* dst) = 0;
    // note: this returns a weak ref
    virtual ID3D12Resource* STDMETHODCALLTYPE GetD3D12ResourceFromAllocation(onnxruntime::IExecutionProvider * provider, void* allocation) = 0;

    // schema overrides (dml does this for us)
    virtual HRESULT STDMETHODCALLTYPE OverrideSchemaInferenceFunctions() = 0;

    // proposed adapter. uses the cross plat ABI currencies
    virtual HRESULT STDMETHODCALLTYPE GetProviderMemoryInfo(onnxruntime::IExecutionProvider * provider, OrtMemoryInfo** memory_info) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetProviderAllocator(onnxruntime::IExecutionProvider * provider, OrtAllocator** allocator) = 0;
    virtual HRESULT STDMETHODCALLTYPE FreeProviderAllocator(OrtAllocator* allocator) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetValueMemoryInfo(const OrtValue * value, OrtMemoryInfo** memory_info) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetMapType(const OrtValue * ort_value, ONNXTensorElementDataType * key_type, ONNXTensorElementDataType * value_type) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVectorMapType(const OrtValue * ort_value, ONNXTensorElementDataType * key_type, ONNXTensorElementDataType * value_type) = 0;
    //virtual HRESULT STDMETHODCALLTYPE CreateTensorFromMap(IInspectable * map, OrtValue * *ort_value) = 0;
    //virtual HRESULT STDMETHODCALLTYPE CreateTensorFromSequence(IInspectable * sequence, OrtValue * *ort_value) = 0;
};

class InferenceSession : public Microsoft::WRL::RuntimeClass <
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    IInferenceSession> {

public:

    InferenceSession(onnxruntime::InferenceSession * session);

    onnxruntime::InferenceSession* STDMETHODCALLTYPE get() noexcept override { 
      return session_.get(); 
    }

    HRESULT STDMETHODCALLTYPE GetOrtSession(OrtSession ** out) noexcept override {
        // (OrtSession *) are really (InferenceSession *) as well
        *out = reinterpret_cast<OrtSession*>(session_.get());
        return S_OK;
    }
    
    void STDMETHODCALLTYPE RegisterGraphTransformers() override;
    HRESULT STDMETHODCALLTYPE RegisterCustomRegistry(IMLOperatorRegistry* registry) override;
    HRESULT STDMETHODCALLTYPE LoadModel(IModelProto* model_proto) override;
    HRESULT STDMETHODCALLTYPE StartProfiling() override;
    HRESULT STDMETHODCALLTYPE EndProfiling() override;
    void STDMETHODCALLTYPE FlushContext(onnxruntime::IExecutionProvider* dml_provider) override;
    void STDMETHODCALLTYPE TrimUploadHeap(onnxruntime::IExecutionProvider* dml_provider) override;
    void STDMETHODCALLTYPE ReleaseCompletedReferences(onnxruntime::IExecutionProvider* dml_provider) override;
    HRESULT STDMETHODCALLTYPE CopyOneInputAcrossDevices(const char* input_name,
                                                        const OrtValue* orig_mlvalue, OrtValue** new_mlvalue) override;


private:
    std::shared_ptr<onnxruntime::InferenceSession> session_;
};

}  // namespace Windows::AI::MachineLearning::Adapter

namespace Ort {
// Ort::Allocator is not in the C ABI yet so it will have to be in the WinMLAdapter for now.
// This struct was copied using the Base struct from onnxruntime_cxx_api.h for reference
// Ort::Allocator struct is used as a smart pointer to OrtAllocator.
struct Allocator {
  Allocator() {
    m_ort_allocator = nullptr;
    m_adapter = nullptr;
  }
  Allocator(winmla::IWinMLAdapter* adapter, OrtAllocator* ort_allocator) :
      m_adapter(adapter), m_ort_allocator(ort_allocator) {}

  ~Allocator() {
    if (m_adapter != nullptr && m_ort_allocator != nullptr) {
      m_adapter->FreeProviderAllocator(m_ort_allocator);
    }
  }

  operator OrtAllocator*() { return m_ort_allocator; }
  operator const OrtAllocator*() const { return m_ort_allocator; }

  OrtAllocator* release() {
    OrtAllocator* p = m_ort_allocator;
    m_ort_allocator = nullptr;
    m_adapter = nullptr;
    return p;
  }

  OrtAllocator** put() noexcept {
    assert(m_ort_allocator == nullptr);
    return &m_ort_allocator;
  }

  Allocator(const Allocator&) = delete;
  Allocator& operator=(const Allocator&) = delete;
  Allocator(Allocator&& v) noexcept :
      m_adapter{v.m_adapter}, m_ort_allocator{v.m_ort_allocator} {
    v.m_adapter = nullptr;
    v.m_ort_allocator = nullptr;
  }
  void operator=(Allocator&& v) noexcept {
    if (m_ort_allocator != nullptr && m_adapter != nullptr) {
      m_adapter->FreeProviderAllocator(m_ort_allocator);
    }
    m_adapter = v.m_adapter;
    m_ort_allocator = v.m_ort_allocator;
    v.m_adapter = nullptr;
    v.m_ort_allocator = nullptr;
  }

 private:
  winmla::IWinMLAdapter* m_adapter;
  OrtAllocator* m_ort_allocator;
};
} // namespace Ort