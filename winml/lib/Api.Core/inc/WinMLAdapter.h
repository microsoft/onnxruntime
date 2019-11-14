// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "IOrtSessionBuilder.h"

namespace Windows::AI::MachineLearning::Adapter {

MIDL_INTERFACE("438e7719-554a-4058-84d9-eb6226c34887") IIOBinding : IUnknown{
    virtual onnxruntime::IOBinding* STDMETHODCALLTYPE get() = 0;
    virtual HRESULT STDMETHODCALLTYPE BindInput(const std::string& name, const OrtValue& ml_value) = 0;
    virtual HRESULT STDMETHODCALLTYPE BindOutput(const std::string& name, const OrtValue& ml_value) = 0;
    virtual const std::vector<std::string>& STDMETHODCALLTYPE GetOutputNames() = 0;
    virtual std::vector<OrtValue>& STDMETHODCALLTYPE GetOutputs() = 0;
};

MIDL_INTERFACE("a848faf6-5a2e-4a7f-b622-cc036f71e28a") IModelProto : IUnknown{
    virtual onnx::ModelProto* STDMETHODCALLTYPE get() = 0;
};

MIDL_INTERFACE("6ec766ef-6365-42bf-b64f-ae85c015adb8") IInferenceSession : IUnknown {
    virtual onnxruntime::InferenceSession* STDMETHODCALLTYPE get() = 0;
    virtual void STDMETHODCALLTYPE RegisterGraphTransformers(bool registerLotusTransforms) = 0;
    virtual HRESULT STDMETHODCALLTYPE RegisterCustomRegistry(IMLOperatorRegistry* registry) = 0;
    virtual HRESULT STDMETHODCALLTYPE LoadModel(IModelProto* model_proto) = 0;
    virtual HRESULT STDMETHODCALLTYPE NewIOBinding(IIOBinding** io_binding) = 0;
    virtual HRESULT STDMETHODCALLTYPE Run(const onnxruntime::RunOptions* run_options, IIOBinding* io_binding) = 0;
    virtual HRESULT STDMETHODCALLTYPE StartProfiling() = 0;
    virtual HRESULT STDMETHODCALLTYPE EndProfiling() = 0;
    virtual void STDMETHODCALLTYPE FlushContext(onnxruntime::IExecutionProvider* dml_provider) = 0;
    virtual void STDMETHODCALLTYPE TrimUploadHeap(onnxruntime::IExecutionProvider* dml_provider) = 0;
    virtual void STDMETHODCALLTYPE ReleaseCompletedReferences(onnxruntime::IExecutionProvider* dml_provider) = 0;
};

// Forward declarations
//namespace onnxruntime {
//    struct SessionOptions;
//    class IExecutionProvider;
//    class InferenceSession;
//}  // namespace onnxruntime

// The IOrtSessionBuilder offers an abstraction over the creation of
// InferenceSession, that enables the creation of the session based on a device (CPU/DML).
MIDL_INTERFACE("2746f03a-7e08-4564-b5d0-c670fef116ee") IOrtSessionBuilder : IUnknown {

  virtual HRESULT STDMETHODCALLTYPE CreateSessionOptions(
      onnxruntime::SessionOptions* options) = 0;

  virtual HRESULT STDMETHODCALLTYPE CreateSession(
      const onnxruntime::SessionOptions& options,
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

    virtual ID3D12Resource* STDMETHODCALLTYPE GetD3D12ResourceFromAllocation(onnxruntime::IExecutionProvider* provider, void* allocation) = 0;

    virtual onnxruntime::Tensor* STDMETHODCALLTYPE CreateTensor(
        winml::TensorKind kind,
        const int64_t * shape,
        uint32_t shape_count,
        onnxruntime::IExecutionProvider* provider) = 0;

    // factory method for creating an ortsessionbuilder from a device
    virtual HRESULT STDMETHODCALLTYPE CreateOrtSessionBuilder(
            ID3D12Device* device,
            ID3D12CommandQueue* queue,
            IOrtSessionBuilder** session_builder) = 0;

    // factory methods for creating an ort model from a path
    virtual HRESULT STDMETHODCALLTYPE CreateModelProto(const char* path, IModelProto** model_proto) = 0;

    // factory methods for creating an ort model from a stream
    virtual HRESULT STDMETHODCALLTYPE CreateModelProto(ABI::Windows::Storage::Streams::IRandomAccessStreamReference* stream_reference, IModelProto** model_proto) = 0;

    // factory methods for creating an ort model from a model_proto
    virtual HRESULT STDMETHODCALLTYPE CreateModelProto(IModelProto * model_proto_in, IModelProto** model_proto) = 0;

    // Data types
    virtual onnxruntime::MLDataType STDMETHODCALLTYPE GetTensorType(winml::TensorKind kind) = 0;
    virtual onnxruntime::MLDataType STDMETHODCALLTYPE GetMapType(winml::TensorKind key_kind, winml::TensorKind value_kind) = 0;
    virtual onnxruntime::MLDataType STDMETHODCALLTYPE GetVectorMapType(winml::TensorKind key_kind, winml::TensorKind value_kind) = 0;
};

extern "C"
__declspec(dllexport) HRESULT STDMETHODCALLTYPE OrtGetWinMLAdapter(IWinMLAdapter** adapter);

class InferenceSession : public Microsoft::WRL::RuntimeClass <
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    IInferenceSession> {

public:

    InferenceSession(onnxruntime::InferenceSession * session);

    onnxruntime::InferenceSession* STDMETHODCALLTYPE get() override { return session_.get(); }
    void STDMETHODCALLTYPE RegisterGraphTransformers(bool registerLotusTransforms) override;
    HRESULT STDMETHODCALLTYPE RegisterCustomRegistry(IMLOperatorRegistry* registry) override;
    HRESULT STDMETHODCALLTYPE LoadModel(IModelProto* model_proto) override;
    HRESULT STDMETHODCALLTYPE NewIOBinding(IIOBinding** io_binding) override;
    HRESULT STDMETHODCALLTYPE Run(const onnxruntime::RunOptions* run_options, IIOBinding* io_binding) override;
    HRESULT STDMETHODCALLTYPE StartProfiling() override;
    HRESULT STDMETHODCALLTYPE EndProfiling() override;
    void STDMETHODCALLTYPE FlushContext(onnxruntime::IExecutionProvider* dml_provider) override;
    void STDMETHODCALLTYPE TrimUploadHeap(onnxruntime::IExecutionProvider* dml_provider) override;
    void STDMETHODCALLTYPE ReleaseCompletedReferences(onnxruntime::IExecutionProvider* dml_provider) override;

private:
    std::shared_ptr<onnxruntime::InferenceSession> session_;
};


// header only code to enable smart pointers on abstract ort objects
template <typename T>
class OrtObject {
 public:
  OrtObject() {
    p_ = nullptr;
  }

  OrtObject(T* m) {
    p_ = m;
  }

  virtual ~OrtObject() {
    if (p_ != nullptr) {
      ReleaseOrtObject(p_);
    }
  }
  T* get() { return p_;  }
private:
  T* p_;
};


}  // namespace Windows::AI::MachineLearning::Adapter