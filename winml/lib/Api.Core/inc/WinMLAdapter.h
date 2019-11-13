// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

namespace Windows::AI::MachineLearning::Adapter {

__declspec(dllexport) HRESULT STDMETHODCALLTYPE ReleaseOrtObject(onnx::ModelProto* model_proto);

__declspec(dllexport) void STDMETHODCALLTYPE EnableDebugOutput();

__declspec(dllexport) HRESULT STDMETHODCALLTYPE EnsureModelDeviceCompatibility(
    winml::LearningModel const& model,
    onnx::ModelProto* p_model_proto,
    bool is_float16_supported);

__declspec(dllexport) ID3D12Resource* STDMETHODCALLTYPE GetD3D12ResourceFromAllocation(onnxruntime::IExecutionProvider* provider, void* allocation);

__declspec(dllexport) onnxruntime::Tensor* STDMETHODCALLTYPE CreateTensor(
    winml::TensorKind kind,
    const int64_t * shape,
    uint32_t shape_count,
    onnxruntime::IExecutionProvider* provider);

class IOBinding
{
public:
    // the RAII smart point stuff that can live in the client dll
    IOBinding(onnxruntime::IOBinding* binding) {
        // take ownership
        binding_ = binding;
    }
    ~IOBinding() {
        this->Release();
    }

    onnxruntime::IOBinding* get() { return binding_; };

    // the rest of this is dll export friendly and can live inside a single implementation
    virtual void STDMETHODCALLTYPE Release();
    virtual HRESULT STDMETHODCALLTYPE BindOutput(const std::string& name, const OrtValue& ml_value);
    virtual const std::vector<std::string>& STDMETHODCALLTYPE GetOutputNames();

private:
    onnxruntime::IOBinding* binding_;
};

class InferenceSession
{
public:
    // the RAII smart point stuff that can live in the client dll
    InferenceSession(onnxruntime::InferenceSession* session) {
        // take ownership
        session_ = session;
    }
    ~InferenceSession() {
        this->Release();
    }

    onnxruntime::InferenceSession* get() { return session_; };

    // the rest of this is dll export friendly and can live inside a single implementation
    virtual void STDMETHODCALLTYPE Release();
    virtual void STDMETHODCALLTYPE RegisterGraphTransformers(bool registerLotusTransforms);
    virtual HRESULT STDMETHODCALLTYPE RegisterCustomRegistry(IMLOperatorRegistry* registry);
    virtual HRESULT STDMETHODCALLTYPE LoadModel(onnx::ModelProto* model_proto);
    virtual HRESULT STDMETHODCALLTYPE NewIOBinding(_winmla::IOBinding** io_binding);
    virtual HRESULT STDMETHODCALLTYPE Run(const onnxruntime::RunOptions* run_options, onnxruntime::IOBinding* io_binding);
    virtual HRESULT STDMETHODCALLTYPE StartProfiling();
    virtual HRESULT STDMETHODCALLTYPE EndProfiling();
    virtual void STDMETHODCALLTYPE FlushContext(onnxruntime::IExecutionProvider* dml_provider);
    virtual void STDMETHODCALLTYPE TrimUploadHeap(onnxruntime::IExecutionProvider* dml_provider);
    virtual void STDMETHODCALLTYPE ReleaseCompletedReferences(onnxruntime::IExecutionProvider* dml_provider);    

private:
    onnxruntime::InferenceSession* session_;
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

using ModelProto = OrtObject<onnx::ModelProto>;

}  // namespace Windows::AI::MachineLearning::Adapter