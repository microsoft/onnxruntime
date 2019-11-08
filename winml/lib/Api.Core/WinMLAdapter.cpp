// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"
#include "inc/WinMLAdapter.h"
#include "inc/CustomRegistryHelper.h"
#include "inc/LotusEnvironment.h"
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"

#include "LearningModelDevice.h"
#include "TensorFeatureDescriptor.h"
#include "ImageFeatureDescriptor.h"
#include "api.image/inc/D3DDeviceCache.h"

using namespace winrt::Windows::AI::MachineLearning;

namespace Windows::AI::MachineLearning::Adapter {

HRESULT STDMETHODCALLTYPE
RegisterCustomRegistry(
    onnxruntime::InferenceSession* p_session,
    IMLOperatorRegistry* registry) {
  RETURN_HR_IF(S_OK, registry == nullptr);
  RETURN_HR_IF_NULL(E_POINTER, p_session);

  auto custom_registries = WinML::GetLotusCustomRegistries(registry);

  // Register
  for (auto& custom_registry : custom_registries) {
    WINML_THROW_IF_NOT_OK(p_session->RegisterCustomRegistry(custom_registry));
  }

  return S_OK;
}

void STDMETHODCALLTYPE EnableDebugOutput() {
  WinML::CWinMLLogSink::EnableDebugOutput();
}

// ORT intentionally requires callers derive from their session class to access
// the protected Load method used below.
class InferenceSessionProtectedLoadAccessor : public onnxruntime::InferenceSession {
 public:
  onnxruntime::common::Status
  Load(std::unique_ptr<ONNX_NAMESPACE::ModelProto> p_model_proto) {
    return onnxruntime::InferenceSession::Load(std::move(p_model_proto));
  }
};

HRESULT STDMETHODCALLTYPE
LoadModel(
    onnxruntime::InferenceSession* session,
    onnx::ModelProto* model_proto) {
  auto session_protected_load_accessor =
      static_cast<InferenceSessionProtectedLoadAccessor*>(session);
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto_ptr(model_proto);
  WINML_THROW_IF_NOT_OK(session_protected_load_accessor->Load(std::move(model_proto_ptr)));
  return S_OK;
}

static bool
IsFeatureDescriptorFp16(
    winml::ILearningModelFeatureDescriptor descriptor) {
  if (auto imageFeatureDescriptor = descriptor.try_as<winmlp::ImageFeatureDescriptor>()) {
    return TensorKind::Float16 == imageFeatureDescriptor->TensorKind();
  }

  if (auto tensorFeatureDescriptor = descriptor.try_as<winmlp::TensorFeatureDescriptor>()) {
    return TensorKind::Float16 == tensorFeatureDescriptor->TensorKind();
  }

  return false;
}

HRESULT STDMETHODCALLTYPE
EnsureModelDeviceCompatibility(
    winml::LearningModel const& model,
    onnx::ModelProto* p_model_proto,
    winml::LearningModelDevice const& device) {
  auto isFloat16Supported = device.as<winmlp::LearningModelDevice>()->GetD3DDeviceCache()->IsFloat16Supported();
  if (!isFloat16Supported) {
    auto& graph = p_model_proto->graph();

    // The model will not contain fp16 operations if:
    // 1. The model has no fp16 inputs
    // 2. The model has no fp16 initializers
    // 3. The model does not create any fp16 intermediary tensors via the Cast (to float16) operator
    // 4. The model does not have any fp16 outputs

    // 1. Ensure that The model has no fp16 inputs
    for (auto descriptor : model.InputFeatures()) {
      WINML_THROW_HR_IF_TRUE_MSG(
          DXGI_ERROR_UNSUPPORTED,
          IsFeatureDescriptorFp16(descriptor),
          "The model contains a 16-bit input (%ls), but the current device does not support 16-bit float.",
          descriptor.Name().c_str());
    }

    // 2. Ensure that the model has no fp16 initializers
    for (int i = 0; i < graph.node_size(); i++) {
      auto node = graph.node(i);
      if (node.op_type() == "Cast" && node.domain().empty()) {
        for (int attribIndex = 0; attribIndex < node.attribute_size(); attribIndex++) {
          auto attribute = node.attribute(attribIndex);
          if (attribute.name() == "to") {
            WINML_THROW_HR_IF_TRUE_MSG(
                DXGI_ERROR_UNSUPPORTED,
                attribute.i() == onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16,
                "The model contains a 16-bit float Cast Op (%s), but the current device does not support 16-bit float.",
                node.name().c_str());
          }
        }
      }
    }

    // 3. Ensure that the model does not create any fp16 intermediary
    //    tensors via the Cast (to float16) operator
    for (int i = 0; i < graph.initializer_size(); i++) {
      auto initializer = graph.initializer(i);

      WINML_THROW_HR_IF_TRUE_MSG(
          DXGI_ERROR_UNSUPPORTED,
          initializer.data_type() == onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16,
          "The model contains a 16-bit float initializer (%s), but the current device does not support 16-bit float.",
          initializer.name().c_str());
    }

    // 4. Ensure that the model does not have any fp16 outputs
    for (auto descriptor : model.OutputFeatures()) {
      WINML_THROW_HR_IF_TRUE_MSG(
          DXGI_ERROR_UNSUPPORTED,
          IsFeatureDescriptorFp16(descriptor),
          "The model contains a 16-bit output (%ls), but the current device does not support 16-bit float.",
          descriptor.Name().c_str());
    }
  }
  return S_OK;
}

HRESULT STDMETHODCALLTYPE ReleaseOrtObject(onnx::ModelProto* model_proto) {
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> ptr(model_proto);
  return S_OK;
}

HRESULT STDMETHODCALLTYPE ReleaseOrtObject(onnxruntime::IOBinding* io_binding) {
  std::unique_ptr<onnxruntime::IOBinding> ptr(io_binding);
  return S_OK;
}

ID3D12Resource* STDMETHODCALLTYPE
GetD3D12ResourceFromAllocation(
    onnxruntime::IExecutionProvider* provider,
    void* allocation) {
  auto d3dResource =
      Dml::GetD3D12ResourceFromAllocation(
          provider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault).get(),
          allocation);
  return d3dResource;
}

onnxruntime::MLDataType GetType(winml::TensorKind kind) {
  switch (kind) {
    case winml::TensorKind::Float:
      return onnxruntime::DataTypeImpl::GetType<float>();
    case winml::TensorKind::Float16:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>();
  };
  return nullptr;
}

onnxruntime::Tensor* STDMETHODCALLTYPE CreateTensor(
    winml::TensorKind kind,
    const int64_t* shape,
    uint32_t shape_count,
    onnxruntime::IExecutionProvider* provider) {
  onnxruntime::TensorShape tensor_shape(shape, shape_count);
  auto pTensor = new onnxruntime::Tensor(
      GetType(kind),
      tensor_shape,
      provider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault));
  return pTensor;
}

}  // namespace Windows::AI::MachineLearning::Adapter