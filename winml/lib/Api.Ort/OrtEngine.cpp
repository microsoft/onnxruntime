#include "pch.h"

#include "OrtEngine.h"
#include "NamespaceAliases.h"

using namespace WinML;

OnnruntimeModel::OnnruntimeModel() : ort_model_(nullptr, nullptr) {
}

STDMETHODIMP OnnruntimeModel::RuntimeClassInitialize(OnnxruntimeEngine* engine, UniqueOrtModel&& ort_model) {
  RETURN_HR_IF_NULL(E_INVALIDARG, ort_model);

  engine_ = engine;
  ort_model_ = std::move(ort_model);

  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetAuthor(const char** out, const size_t* len) {
  auto winml_adapter_api = engine_->UseWinmlAdapterApi();
  if (auto status = winml_adapter_api->ModelGetAuthor(ort_model_.get(), out, len)) {
    return E_FAIL;
  }
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetName(const char** out, const size_t* len) {
  auto winml_adapter_api = engine_->UseWinmlAdapterApi();
  if (auto status = winml_adapter_api->ModelGetName(ort_model_.get(), out, len)) {
    return E_FAIL;
  }
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetDomain(const char** out, const size_t* len) {
  auto winml_adapter_api = engine_->UseWinmlAdapterApi();
  if (auto status = winml_adapter_api->ModelGetDomain(ort_model_.get(), out, len)) {
    return E_FAIL;
  }
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetDescription(const char** out, const size_t* len) {
  auto winml_adapter_api = engine_->UseWinmlAdapterApi();
  if (auto status = winml_adapter_api->ModelGetDescription(ort_model_.get(), out, len)) {
    return E_FAIL;
  }
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetVersion(int64_t* out) {
  auto winml_adapter_api = engine_->UseWinmlAdapterApi();
  if (auto status = winml_adapter_api->ModelGetVersion(ort_model_.get(), out)) {
    return E_FAIL;
  }
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetModelMetadata(ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING>** metadata) {
  return E_NOTIMPL;
}

STDMETHODIMP OnnruntimeModel::GetInputFeatures(ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features) {
  return E_NOTIMPL;
}

STDMETHODIMP OnnruntimeModel::GetOutputFeatures(ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features) {
  return E_NOTIMPL;
}

STDMETHODIMP OnnruntimeModel::CloneModel(IModel** copy) {
  return E_NOTIMPL;
}

HRESULT OnnxruntimeEngine::RuntimeClassInitialize() {
  const uint32_t ort_version = 1;
  const auto ort_api_base = OrtGetApiBase();
  ort_api_ = ort_api_base->GetApi(ort_version);
  winml_adapter_api_ = GetWinmlAdapterApi(ort_api_);
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngine::CreateModel(_In_ const char* model_path, _In_ size_t len, _Outptr_ IModel** out) {
  OrtModel* ort_model = nullptr;
  if (auto status = winml_adapter_api_->CreateModelFromPath(model_path, len, &ort_model)) {
    return E_FAIL;
  }

  auto model = UniqueOrtModel(ort_model, winml_adapter_api_->ReleaseModel);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnruntimeModel>(out, this, std::move(model)));
  return E_NOTIMPL;
}

STDMETHODIMP OnnxruntimeEngine::CreateModel(_In_ void* data, _In_ size_t size, _Outptr_ IModel** out) {
  OrtModel* ort_model = nullptr;
  if (auto status = winml_adapter_api_->CreateModelFromData(data, size, &ort_model)) {
    return E_FAIL;
  }

  auto model = UniqueOrtModel(ort_model, winml_adapter_api_->ReleaseModel);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnruntimeModel>(out, this, std::move(model)));
  return E_NOTIMPL;
}

const OrtApi* OnnxruntimeEngine::UseOrtApi() {
  return ort_api_;
}

const WinmlAdapterApi* OnnxruntimeEngine::UseWinmlAdapterApi() {
  return winml_adapter_api_;
}

STDAPI CreateOrtEngine(_Out_ IEngine** engine) {
  auto onnxruntime_engine = Microsoft::WRL::Make<Windows::AI::MachineLearning::OnnxruntimeEngine>();
  onnxruntime_engine.CopyTo(engine);
  return E_NOTIMPL;
}