// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

namespace Windows::AI::MachineLearning::Adapter {

__declspec(dllexport) HRESULT STDMETHODCALLTYPE ReleaseOrtObject(onnx::ModelProto* model_proto);
__declspec(dllexport) HRESULT STDMETHODCALLTYPE ReleaseOrtObject(onnxruntime::IOBinding* io_binding);

__declspec(dllexport) HRESULT STDMETHODCALLTYPE RegisterCustomRegistry(
    onnxruntime::InferenceSession* p_session,
    IMLOperatorRegistry* registry);

__declspec(dllexport) void STDMETHODCALLTYPE EnableDebugOutput();

__declspec(dllexport) HRESULT STDMETHODCALLTYPE LoadModel(onnxruntime::InferenceSession* session, onnx::ModelProto* model);

__declspec(dllexport) HRESULT STDMETHODCALLTYPE EnsureModelDeviceCompatibility(
    winml::LearningModel const& model,
    onnx::ModelProto* p_model_proto,
    winml::LearningModelDevice const& device);

__declspec(dllexport) ID3D12Resource* STDMETHODCALLTYPE GetD3D12ResourceFromAllocation(onnxruntime::IExecutionProvider* provider, void* allocation);

__declspec(dllexport) onnxruntime::Tensor* STDMETHODCALLTYPE CreateTensor(
    winml::TensorKind kind,
    const int64_t * shape,
    uint32_t shape_count,
    onnxruntime::IExecutionProvider* provider);

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

  T* p_;
};
using ModelProto = OrtObject<onnx::ModelProto>;
using IOBinding = OrtObject<onnxruntime::IOBinding>;

}  // namespace Windows::AI::MachineLearning::Adapter