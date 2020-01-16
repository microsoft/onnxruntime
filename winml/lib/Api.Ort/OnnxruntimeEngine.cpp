#include "pch.h"

#include "OnnxruntimeEngine.h"

#include "PheonixSingleton.h"
#include "OnnxruntimeEnvironment.h"
#include "OnnxruntimeEngineBuilder.h"
#include "OnnxruntimeModel.h"
#include "OnnxruntimeSessionBuilder.h"

// Add back when we remove the winmladapter.h
//#include "core/providers/winml/winml_provider_factory.h"

using namespace WinML;

OnnxruntimeEngine::OnnxruntimeEngine() : session_(nullptr, nullptr), provider_(nullptr, nullptr) {
}

HRESULT OnnxruntimeEngine::RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory,
                                                  UniqueOrtSession&& session,
                                                  IOrtSessionBuilder* session_builder) {
  engine_factory_ = engine_factory;
  session_ = std::move(session);
  session_builder_ = session_builder;
  return S_OK;
}

HRESULT OnnxruntimeEngine::LoadModel(_In_ IModel* model) {
  Microsoft::WRL::ComPtr<IOnnxruntimeModel> onnxruntime_model;
  RETURN_IF_FAILED(model->QueryInterface(IID_PPV_ARGS(&onnxruntime_model)));

  OrtModel* ort_model;
  RETURN_IF_FAILED(onnxruntime_model->DetachOrtModel(&ort_model));

  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  winml_adapter_api->SessionLoadAndPurloinModel(session_.get(), ort_model);

  return S_OK;
}

HRESULT OnnxruntimeEngine::Initialize() {
  RETURN_IF_FAILED(session_builder_->Initialize(session_.get()));
  return S_OK;
}

HRESULT OnnxruntimeEngine::RegisterGraphTransformers() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  winml_adapter_api->SessionRegisterGraphTransformers(session_.get());
  return S_OK;
}

HRESULT OnnxruntimeEngine::RegisterCustomRegistry(IMLOperatorRegistry* registry) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  winml_adapter_api->SessionRegisterCustomRegistry(session_.get(), registry);  
  return S_OK;
}

HRESULT OnnxruntimeEngine::EndProfiling() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  winml_adapter_api->SessionEndProfiling(session_.get());
  return S_OK;
}

HRESULT OnnxruntimeEngine::StartProfiling() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtEnv* ort_env;
  engine_factory_->GetOrtEnvironment(&ort_env);

  winml_adapter_api->SessionStartProfiling(ort_env, session_.get());
  return S_OK;
}

HRESULT OnnxruntimeEngine::FlushContext() {
  return E_NOTIMPL;
}

HRESULT OnnxruntimeEngine::TrimUploadHeap() {
  return E_NOTIMPL;
}

HRESULT OnnxruntimeEngine::ReleaseCompletedReferences() {
  return E_NOTIMPL;
}

HRESULT OnnxruntimeEngine::CopyOneInputAcrossDevices(const char* input_name, const IValue* src, IValue** dest) {
  return E_NOTIMPL;
}

HRESULT OnnxruntimeEngine::Sync() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  const OrtExecutionProvider* ort_provider;
  winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider);

  winml_adapter_api->ExecutionProviderSync(ort_provider);
  return S_OK;
}

HRESULT OnnxruntimeEngine::CreateTensorValue(int64_t* /*shape*/, size_t /*count*/, winml::TensorKind /*kind*/, _Out_ IValue** /*out*/) {
  return E_NOTIMPL;
}

HRESULT OnnxruntimeEngine::CopyOneInputAcrossDevices(const char* name, IValue* src, IValue** out) {
  return E_NOTIMPL;
}

// TODO supposedly this doesnt work if it is not static
static std::shared_ptr<OnnxruntimeEnvironment> onnxruntime_environment_;

HRESULT OnnxruntimeEngineFactory::RuntimeClassInitialize() {
  const uint32_t ort_version = 1;
  const auto ort_api_base = OrtGetApiBase();
  ort_api_ = ort_api_base->GetApi(ort_version);
  winml_adapter_api_ = GetWinmlAdapterApi(ort_api_);

  environment_ = onnxruntime_environment_ = PheonixSingleton<OnnxruntimeEnvironment>(ort_api_);
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineFactory::CreateModel(_In_ const char* model_path, _In_ size_t len, _Outptr_ IModel** out) {
  OrtModel* ort_model = nullptr;
  if (auto status = winml_adapter_api_->CreateModelFromPath(model_path, len, &ort_model)) {
    return E_FAIL;
  }

  auto model = UniqueOrtModel(ort_model, winml_adapter_api_->ReleaseModel);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnruntimeModel>(out, this, std::move(model)));
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineFactory::CreateModel(_In_ void* data, _In_ size_t size, _Outptr_ IModel** out) {
  OrtModel* ort_model = nullptr;
  if (auto status = winml_adapter_api_->CreateModelFromData(data, size, &ort_model)) {
    return E_FAIL;
  }

  auto model = UniqueOrtModel(ort_model, winml_adapter_api_->ReleaseModel);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnruntimeModel>(out, this, std::move(model)));
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineFactory::CreateEngineBuilder(_Outptr_ Windows::AI::MachineLearning::IEngineBuilder** out) {
  Microsoft::WRL::ComPtr<OnnxruntimeEngineBuilder> onnxruntime_engine_builder;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeEngineBuilder>(&onnxruntime_engine_builder, this));
  RETURN_IF_FAILED(onnxruntime_engine_builder.CopyTo(out));
  return S_OK;
}

const OrtApi* OnnxruntimeEngineFactory::UseOrtApi() {
  return ort_api_;
}

const WinmlAdapterApi* OnnxruntimeEngineFactory::UseWinmlAdapterApi() {
  return winml_adapter_api_;
}

HRESULT OnnxruntimeEngineFactory::GetOrtEnvironment(OrtEnv** ort_env) {
  RETURN_IF_FAILED(environment_->GetOrtEnvironment(ort_env));
  return S_OK;
}

STDAPI CreateOnnxruntimeEngineFactory(_Out_ Windows::AI::MachineLearning::IEngineFactory** engine_factory) {
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> onnxruntime_engine_factory;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeEngineFactory>(&onnxruntime_engine_factory));
  RETURN_IF_FAILED(onnxruntime_engine_factory.CopyTo(engine_factory));
  return S_OK;
}

/* add these implementation pieces into the right places into the onnxruntime value/engine api calls

engine->CreateValue

  Ort::Allocator dml_allocator(m_adapter.get(), nullptr);
  WINML_THROW_IF_FAILED(m_adapter->GetProviderAllocator(provider, dml_allocator.put())); get from engine

  // create the OrtValue as a tensor letting ort know that we own the data buffer
  Ort::Value ort_tensor = Ort::Value::CreateTensor(
      dml_allocator,
      &(resourceMetadata.TensorDescriptor.sizes[0]),
      sizeof(resourceMetadata.TensorDescriptor.sizes) / sizeof(resourceMetadata.TensorDescriptor.sizes[0]),
      (resourceMetadata.TensorDescriptor.dataType == kImageTensorDataTypeFloat32) ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

value->GetResource

  void* pAllocatedResource = nullptr;
  Ort::ThrowOnError(Ort::GetApi().GetTensorMutableData(ort_tensor, &pAllocatedResource));

  if (dml)
      auto d3dResource =
          adapter->GetD3D12ResourceFromAllocation(
              spSession->GetExecutionProvider(),
              allocated_resource);

value->IsCpu()
    //Ort::MemoryInfo memory_info(nullptr);
    //m_adapter->GetValueMemoryInfo(ort_value, memory_info.put());

    if (!strcmp(memory_info.Name(), onnxruntime::CPU) ||
        memory_info.MemType() == ::OrtMemType::OrtMemTypeCPUOutput ||
        memory_info.MemType() == ::OrtMemType::OrtMemTypeCPUInput) {
  bool LearningModelBinding::IsOfMapType(const Ort::Value& ort_value, TensorKind key_kind, TensorKind value_kind) {
    if (ort_value.GetTypeInfo().GetONNXType() != ONNX_TYPE_MAP)
      return false;

    ONNXTensorElementDataType onnx_key_type;
    ONNXTensorElementDataType onnx_value_type;

    WINML_THROW_IF_FAILED(adapter_->GetMapType(ort_value, &onnx_key_type, &onnx_value_type));

    if (onnx_key_type != GetONNXTensorElementDataType(key_kind))
      return false;

    if (onnx_value_type != GetONNXTensorElementDataType(value_kind))
      return false;

    return true;
  };

  bool LearningModelBinding::IsOfVectorMapType(const Ort::Value& ort_value, TensorKind key_kind, TensorKind value_kind) {
    if (ort_value.GetTypeInfo().GetONNXType() != ONNX_TYPE_SEQUENCE)
      return false;

    ONNXTensorElementDataType onnx_key_type;
    ONNXTensorElementDataType onnx_value_type;

    WINML_THROW_IF_FAILED(adapter_->GetVectorMapType(ort_value, &onnx_key_type, &onnx_value_type));

    if (onnx_key_type != GetONNXTensorElementDataType(key_kind))
      return false;

    if (onnx_value_type != GetONNXTensorElementDataType(value_kind))
      return false;

    return true;
  };

  bool LearningModelBinding::IsOfTensorType(const Ort::Value& ort_value, TensorKind kind) {
    return ort_value.GetTensorTypeAndShapeInfo().GetElementType() == GetONNXTensorElementDataType(kind);
  };

  gettensorshape
      uint32_t width = static_cast<uint32_t>(ort_value.GetTensorTypeAndShapeInfo().GetShape()[3]);
  uint32_t height = static_cast<uint32_t>(ort_value.GetTensorTypeAndShapeInfo().GetShape()[2]);
  uint32_t batchSize = static_cast<uint32_t>(ort_value.GetTensorTypeAndShapeInfo().GetShape()[0]);

  */