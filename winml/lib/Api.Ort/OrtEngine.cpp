#include "pch.h"

#include "OrtEngine.h"
#include "NamespaceAliases.h"
#include "FeatureDescriptorFactory.h"

// Add back when we remove the winmladapter.h
//#include "core/providers/winml/winml_provider_factory.h"

using namespace WinML;

struct feature_helper_functions {
  using FnGetCount = decltype(WinmlAdapterApi::ModelGetInputCount);
  FnGetCount GetCount;

  using FnGetName = decltype(WinmlAdapterApi::ModelGetInputName);
  FnGetName GetName;

  using FnGetDescription = decltype(WinmlAdapterApi::ModelGetInputDescription);
  FnGetDescription GetDescription;

  using FnGetTypeInfo = decltype(WinmlAdapterApi::ModelGetInputTypeInfo);
  FnGetTypeInfo GetTypeInfo;
};

HRESULT CreateFeatureDescriptors(
    const feature_helper_functions* feature_helpers,
    OrtModel* ort_model,
    std::vector<FeatureDescriptor>& descriptors) {
  size_t count;
  if (auto status = feature_helpers->GetCount(ort_model, &count)) {
    return E_FAIL;
  }

  for (size_t i = 0; i < count; i++) {
    FeatureDescriptor descriptor;
    if (auto status = feature_helpers->GetName(ort_model, i, &descriptor.name_, &descriptor.name_length_)) {
      return E_FAIL;
    }
    if (auto status = feature_helpers->GetDescription(ort_model, i, &descriptor.description, &descriptor.description_length_)) {
      return E_FAIL;
    }
    if (auto status = feature_helpers->GetTypeInfo(ort_model, i, &descriptor.type_info_)) {
      return E_FAIL;
    }

    descriptors.push_back(descriptor);
  }
  return S_OK;
}

HRESULT ModelInfo::RuntimeClassInitialize(OnnxruntimeEngine* engine, OrtModel* ort_model) {
  RETURN_HR_IF_NULL(E_INVALIDARG, ort_model);

  const auto winml_adapter_api = engine->UseWinmlAdapterApi();

  // Get Metadata
  size_t count;
  if (auto status = winml_adapter_api->ModelGetMetadataCount(ort_model, &count)) {
    return E_FAIL;
  }

  const char* metadata_key;
  size_t metadata_key_len;
  const char* metadata_value;
  size_t metadata_value_len;
  for (size_t i = 0; i < count; i++) {
    if (auto status = winml_adapter_api->ModelGetMetadata(ort_model, count, &metadata_key, &metadata_key_len, &metadata_value, &metadata_value_len)) {
      return E_FAIL;
    }

    model_metadata_.insert_or_assign(std::string(metadata_key, metadata_key_len), std::string(metadata_value, metadata_value_len));
  }

  WinML::FeatureDescriptorFactory builder(model_metadata_);

  static const feature_helper_functions input_helpers = {
      winml_adapter_api->ModelGetInputCount,
      winml_adapter_api->ModelGetInputName,
      winml_adapter_api->ModelGetInputDescription,
      winml_adapter_api->ModelGetInputTypeInfo
  };

  // Create inputs
  std::vector<FeatureDescriptor> inputs;
  RETURN_IF_FAILED(CreateFeatureDescriptors(&input_helpers, ort_model, inputs));
  input_features_ = builder.CreateLearningModelFeatureDescriptors(inputs);

  // Create outputs
  static const feature_helper_functions output_helpers = {
      winml_adapter_api->ModelGetOutputCount,
      winml_adapter_api->ModelGetOutputName,
      winml_adapter_api->ModelGetOutputDescription,
      winml_adapter_api->ModelGetOutputTypeInfo};

  std::vector<FeatureDescriptor> outputs;
  RETURN_IF_FAILED(CreateFeatureDescriptors(&output_helpers, ort_model, outputs));
  output_features_ = builder.CreateLearningModelFeatureDescriptors(outputs);

  const char* out;
  size_t len;

  if (auto status = winml_adapter_api->ModelGetAuthor(ort_model, &out, &len)) {
    return E_FAIL;
  }
  author_ = std::string(out, len);

  if (auto status = winml_adapter_api->ModelGetName(ort_model, &out, &len)) {
    return E_FAIL;
  }
  name_ = std::string(out, len);

  if (auto status = winml_adapter_api->ModelGetDomain(ort_model, &out, &len)) {
    return E_FAIL;
  }
  domain_ = std::string(out, len);

  if (auto status = winml_adapter_api->ModelGetDescription(ort_model, &out, &len)) {
    return E_FAIL;
  }
  description_ = std::string(out, len);

  if (auto status = winml_adapter_api->ModelGetVersion(ort_model, &version_)) {
    return E_FAIL;
  }

  return S_OK;
}

STDMETHODIMP ModelInfo::GetAuthor(const char** out, size_t* len) {
  *out = author_.c_str();
  *len = author_.size();
  return S_OK;
}

STDMETHODIMP ModelInfo::GetName(const char** out, size_t* len) {
  *out = name_.c_str();
  *len = name_.size();
  return S_OK;
}

STDMETHODIMP ModelInfo::GetDomain(const char** out, size_t* len) {
  *out = domain_.c_str();
  *len = domain_.size();
  return S_OK;
}

STDMETHODIMP ModelInfo::GetDescription(const char** out, size_t* len) {
  *out = description_.c_str();
  *len = description_.size();
  return S_OK;
}

STDMETHODIMP ModelInfo::GetVersion(int64_t* out) {
  *out = version_;
  return S_OK;
}

STDMETHODIMP ModelInfo::GetModelMetadata(ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING>** metadata) {
  std::unordered_map<winrt::hstring, winrt::hstring> map_copy;
  for (auto& pair : model_metadata_) {
    auto metadata_key = WinML::Strings::HStringFromUTF8(pair.first);
    auto metadata_value = WinML::Strings::HStringFromUTF8(pair.second);
    map_copy.emplace(std::move(metadata_key), std::move(metadata_value));
  }
  auto map = winrt::single_threaded_map<winrt::hstring, winrt::hstring>(std::move(map_copy));
  winrt::attach_abi(map, *metadata);
  return S_OK;
}

STDMETHODIMP ModelInfo::GetInputFeatures(ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features) {
  *features = nullptr;
  winrt::copy_to_abi(input_features_.GetView(), *(void**)features);
  return S_OK;
}

STDMETHODIMP ModelInfo::GetOutputFeatures(ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features) {
  *features = nullptr;
  winrt::copy_to_abi(output_features_.GetView(), *(void**)features);
  return E_NOTIMPL;
}

OnnruntimeModel::OnnruntimeModel() : ort_model_(nullptr, nullptr) {
}

STDMETHODIMP OnnruntimeModel::RuntimeClassInitialize(OnnxruntimeEngine* engine, UniqueOrtModel&& ort_model) {
  RETURN_HR_IF_NULL(E_INVALIDARG, ort_model);

  engine_ = engine;
  ort_model_ = std::move(ort_model);

  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetModelInfo(IModelInfo** info) {
  if (info_ == nullptr) {
    RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<ModelInfo>(&info_, engine_.Get(), ort_model_.get()));
  }

  info_.CopyTo(info);

  return S_OK;
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