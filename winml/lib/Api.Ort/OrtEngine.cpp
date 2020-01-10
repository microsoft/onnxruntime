#include "pch.h"

#include "OrtEngine.h"
#include "NamespaceAliases.h"
#include "FeatureDescriptorFactory.h"

// Add back when we remove the winmladapter.h
//#include "core/providers/winml/winml_provider_factory.h"

using namespace WinML;

OnnruntimeModel::OnnruntimeModel() : ort_model_(nullptr, nullptr) {
}

STDMETHODIMP OnnruntimeModel::RuntimeClassInitialize(OnnxruntimeEngine* engine, UniqueOrtModel&& ort_model) {
  RETURN_HR_IF_NULL(E_INVALIDARG, ort_model);

  engine_ = engine;
  ort_model_ = std::move(ort_model);

  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetAuthor(const char** out, size_t* len) {
  auto winml_adapter_api = engine_->UseWinmlAdapterApi();
  if (auto status = winml_adapter_api->ModelGetAuthor(ort_model_.get(), out, len)) {
    return E_FAIL;
  }
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetName(const char** out, size_t* len) {
  auto winml_adapter_api = engine_->UseWinmlAdapterApi();
  if (auto status = winml_adapter_api->ModelGetName(ort_model_.get(), out, len)) {
    return E_FAIL;
  }
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetDomain(const char** out, size_t* len) {
  auto winml_adapter_api = engine_->UseWinmlAdapterApi();
  if (auto status = winml_adapter_api->ModelGetDomain(ort_model_.get(), out, len)) {
    return E_FAIL;
  }
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetDescription(const char** out, size_t* len) {
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

HRESULT OnnruntimeModel::EnsureMetadata() {
  if (metadata_cache_.has_value() == false) {
    auto winml_adapter_api = engine_->UseWinmlAdapterApi();

    size_t count;
    if (auto status = winml_adapter_api->ModelGetMetadataCount(ort_model_.get(), &count)) {
      return E_FAIL;
    }

    std::unordered_map<std::string, std::string> metadata;

    const char* metadata_key;
    size_t metadata_key_len;
    const char* metadata_value;
    size_t metadata_value_len;
    for (size_t i = 0; i < count; i++) {
      if (auto status = winml_adapter_api->ModelGetMetadata(ort_model_.get(), count, &metadata_key, &metadata_key_len, &metadata_value, &metadata_value_len)) {
        return E_FAIL;
      }
      metadata.insert_or_assign(std::string(metadata_key, metadata_key_len), std::string(metadata_value, metadata_value_len));
    }

    metadata_cache_ = std::move(metadata);
  }

  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetModelMetadata(ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING>** metadata) {
  RETURN_IF_FAILED(EnsureMetadata());

  std::unordered_map<winrt::hstring, winrt::hstring> map_copy;
  for (auto& pair : metadata_cache_.value()) {
    auto metadata_key = WinML::Strings::HStringFromUTF8(pair.first);
    auto metadata_value = WinML::Strings::HStringFromUTF8(pair.second);
    map_copy.emplace(std::move(metadata_key), std::move(metadata_value));
  }
  auto map = winrt::single_threaded_map<winrt::hstring, winrt::hstring>(std::move(map_copy));
  winrt::attach_abi(map, *metadata);
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetInputFeatures(ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features) {
  RETURN_IF_FAILED(EnsureMetadata());
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetOutputFeatures(ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features) {
  RETURN_IF_FAILED(EnsureMetadata());
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

static HRESULT OverrideSchemaInferenceFunctions() {
  // This only makes sense for ORT.
  // Before creating any models, we ensure that the schema has been overridden.
  // TODO... need to call into the appro
  //WINML_THROW_IF_FAILED(adapter_->OverrideSchemaInferenceFunctions());
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngine::CreateModel(_In_ const char* model_path, _In_ size_t len, _Outptr_ IModel** out) {
  OverrideSchemaInferenceFunctions();

  OrtModel* ort_model = nullptr;
  if (auto status = winml_adapter_api_->CreateModelFromPath(model_path, len, &ort_model)) {
    return E_FAIL;
  }

  auto model = UniqueOrtModel(ort_model, winml_adapter_api_->ReleaseModel);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnruntimeModel>(out, this, std::move(model)));
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngine::CreateModel(_In_ void* data, _In_ size_t size, _Outptr_ IModel** out) {
  OverrideSchemaInferenceFunctions();

  OrtModel* ort_model = nullptr;
  if (auto status = winml_adapter_api_->CreateModelFromData(data, size, &ort_model)) {
    return E_FAIL;
  }

  auto model = UniqueOrtModel(ort_model, winml_adapter_api_->ReleaseModel);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnruntimeModel>(out, this, std::move(model)));
  return S_OK;
}

const OrtApi* OnnxruntimeEngine::UseOrtApi() {
  return ort_api_;
}

const WinmlAdapterApi* OnnxruntimeEngine::UseWinmlAdapterApi() {
  return winml_adapter_api_;
}

STDAPI CreateOrtEngine(_Out_ IEngine** engine) {
  Microsoft::WRL::ComPtr<Windows::AI::MachineLearning::OnnxruntimeEngine> onnxruntime_engine;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<Windows::AI::MachineLearning::OnnxruntimeEngine>(&onnxruntime_engine));
  RETURN_IF_FAILED(onnxruntime_engine.CopyTo(engine));
  return S_OK;
}