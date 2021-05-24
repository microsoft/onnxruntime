// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"
#include "OnnxruntimeModel.h"
#include "core/platform/windows/TraceLoggingConfig.h"
#include <evntrace.h>

#include "OnnxruntimeDescriptorConverter.h"
#include "OnnxruntimeEngine.h"
#include "OnnxruntimeErrors.h"

using namespace _winml;

struct winml_adapter_api_model_feature_helper {
  decltype(WinmlAdapterApi::ModelGetInputCount) GetCount;
  decltype(WinmlAdapterApi::ModelGetInputName) GetName;
  decltype(WinmlAdapterApi::ModelGetInputDescription) GetDescription;
  decltype(WinmlAdapterApi::ModelGetInputTypeInfo) GetTypeInfo;
};

HRESULT CreateFeatureDescriptors(
    OnnxruntimeEngineFactory* engine_factory,
    const winml_adapter_api_model_feature_helper* feature_helpers,
    OrtModel* ort_model,
    std::vector<OnnxruntimeValueInfoWrapper>& descriptors) {
  const auto ort_api = engine_factory->UseOrtApi();
  size_t count;
  RETURN_HR_IF_NOT_OK_MSG(feature_helpers->GetCount(ort_model, &count),
                          engine_factory->UseOrtApi());

  for (size_t i = 0; i < count; i++) {
    OnnxruntimeValueInfoWrapper descriptor;
    RETURN_HR_IF_NOT_OK_MSG(feature_helpers->GetName(ort_model, i, &descriptor.name_, &descriptor.name_length_),
                            engine_factory->UseOrtApi());
    RETURN_HR_IF_NOT_OK_MSG(feature_helpers->GetDescription(ort_model, i, &descriptor.description_, &descriptor.description_length_),
                            engine_factory->UseOrtApi());

    OrtTypeInfo* type_info;
    RETURN_HR_IF_NOT_OK_MSG(feature_helpers->GetTypeInfo(ort_model, i, &type_info),
                            engine_factory->UseOrtApi());

    descriptor.type_info_ = UniqueOrtTypeInfo(type_info, ort_api->ReleaseTypeInfo);

    descriptors.push_back(std::move(descriptor));
  }
  return S_OK;
}

HRESULT ModelInfo::RuntimeClassInitialize(_In_ OnnxruntimeEngineFactory* engine_factory, _In_ OrtModel* ort_model) {
  RETURN_HR_IF_NULL(E_INVALIDARG, ort_model);

  const auto winml_adapter_api = engine_factory->UseWinmlAdapterApi();

  // Get Metadata
  size_t count;
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelGetMetadataCount(ort_model, &count),
                          engine_factory->UseOrtApi());

  const char* metadata_key;
  size_t metadata_key_len;
  const char* metadata_value;
  size_t metadata_value_len;
  for (size_t i = 0; i < count; i++) {
    RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelGetMetadata(ort_model, i, &metadata_key, &metadata_key_len, &metadata_value, &metadata_value_len),
                            engine_factory->UseOrtApi());

    model_metadata_.insert_or_assign(
        std::string(metadata_key, metadata_key_len),
        std::string(metadata_value, metadata_value_len));
  }

  _winml::OnnxruntimeDescriptorConverter converter(engine_factory, model_metadata_);

  static const winml_adapter_api_model_feature_helper input_helpers = {
      winml_adapter_api->ModelGetInputCount,
      winml_adapter_api->ModelGetInputName,
      winml_adapter_api->ModelGetInputDescription,
      winml_adapter_api->ModelGetInputTypeInfo};

  // Create inputs
  std::vector<OnnxruntimeValueInfoWrapper> inputs;
  RETURN_IF_FAILED(CreateFeatureDescriptors(engine_factory, &input_helpers, ort_model, inputs));
  input_features_ = converter.ConvertToLearningModelDescriptors(inputs.data(), inputs.size());

  // Create outputs
  static const winml_adapter_api_model_feature_helper output_helpers = {
      winml_adapter_api->ModelGetOutputCount,
      winml_adapter_api->ModelGetOutputName,
      winml_adapter_api->ModelGetOutputDescription,
      winml_adapter_api->ModelGetOutputTypeInfo};

  std::vector<OnnxruntimeValueInfoWrapper> outputs;
  RETURN_IF_FAILED(CreateFeatureDescriptors(engine_factory, &output_helpers, ort_model, outputs));
  output_features_ = converter.ConvertToLearningModelDescriptors(outputs.data(), outputs.size());

  const char* out;
  size_t len;

  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelGetAuthor(ort_model, &out, &len),
                          engine_factory->UseOrtApi());
  author_ = std::string(out, len);

  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelGetName(ort_model, &out, &len),
                          engine_factory->UseOrtApi());
  name_ = std::string(out, len);

  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelGetDomain(ort_model, &out, &len),
                          engine_factory->UseOrtApi());
  domain_ = std::string(out, len);

  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelGetDescription(ort_model, &out, &len),
                          engine_factory->UseOrtApi());
  description_ = std::string(out, len);

  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelGetVersion(ort_model, &version_),
                          engine_factory->UseOrtApi());
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

struct CaseInsensitiveHash {
  size_t operator()(const winrt::hstring& key) const {
    size_t h = 0, i = 0;
    std::for_each(key.begin(), key.end(), [&](wchar_t c) {
      i++;
      h += i * towlower(c);
    });
    return h;
  }
};

struct CaseInsensitiveEqual {
  bool operator()(const winrt::hstring& left, const winrt::hstring& right) const {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(),
                                                     [](wchar_t a, wchar_t b) {
                                                       return towlower(a) == towlower(b);
                                                     });
  }
};

STDMETHODIMP ModelInfo::GetModelMetadata(ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING>** metadata) {
  std::unordered_map<winrt::hstring, winrt::hstring, CaseInsensitiveHash, CaseInsensitiveEqual> map_copy;
  for (auto& pair : model_metadata_) {
    auto metadata_key = _winml::Strings::HStringFromUTF8(pair.first);
    auto metadata_value = _winml::Strings::HStringFromUTF8(pair.second);
    map_copy.emplace(std::move(metadata_key), std::move(metadata_value));
  }
  auto map = winrt::single_threaded_map<winrt::hstring, winrt::hstring>(std::move(map_copy));
  winrt::copy_to_abi(map, *(void**)metadata);
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
  return S_OK;
}

OnnruntimeModel::OnnruntimeModel() : ort_model_(nullptr, nullptr) {
}

HRESULT OnnruntimeModel::RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory, UniqueOrtModel&& ort_model) {
  RETURN_HR_IF_NULL(E_INVALIDARG, ort_model);

  engine_factory_ = engine_factory;
  ort_model_ = std::move(ort_model);

  return S_OK;
}

STDMETHODIMP OnnruntimeModel::GetModelInfo(IModelInfo** info) {
  if (info_ == nullptr) {
    RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<ModelInfo>(&info_, engine_factory_.Get(), ort_model_.get()));
  }

  info_.CopyTo(info);

  return S_OK;
}

STDMETHODIMP OnnruntimeModel::ModelEnsureNoFloat16() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  if (auto status = winml_adapter_api->ModelEnsureNoFloat16(ort_model_.get())) {
    return DXGI_ERROR_UNSUPPORTED;
  }
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::CloneModel(IModel** copy) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtModel* ort_model_copy;
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->CloneModel(ort_model_.get(), &ort_model_copy),
                          engine_factory_->UseOrtApi());

  auto model = UniqueOrtModel(ort_model_copy, winml_adapter_api->ReleaseModel);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnruntimeModel>(copy, engine_factory_.Get(), std::move(model)));

  return S_OK;
}

STDMETHODIMP OnnruntimeModel::SaveModel(_In_ const wchar_t* const file_name, _In_ unsigned size) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->SaveModel(ort_model_.get(), file_name, size),
                          engine_factory_->UseOrtApi());
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::DetachOrtModel(OrtModel** model) {
  *model = ort_model_.release();
  return S_OK;
}

HRESULT GetValue(const char* key, const char* const* keys, const char* const* values,
                 size_t num_values_in_dictionary, const char** value) {
  auto found_it =
      std::find_if(keys, keys + num_values_in_dictionary, [key](auto& key_name) {
        return _stricmp(key, key_name) == 0;
      });
  if (found_it == (keys + num_values_in_dictionary)) {
    return S_FALSE;
  }
  *value = values[std::distance(keys, found_it)];
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::AddOperator(
    _In_ const char* const op_type, _In_ const char* const op_name, _In_ const char* const op_domain,
    _In_ const char* const* op_input_names, _In_ const char* const* actual_input_names, size_t num_inputs,
    _In_ const char* const* op_output_names, _In_ const char* const* actual_output_names, size_t num_outputs,
    _In_ const char* const* op_attribute_names, _In_ IValue** attribute_values, size_t num_attributes) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  auto ort_api = engine_factory_->UseOrtApi();

  int32_t onnx_opset_version;
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelGetOpsetVersion(ort_model_.get(), op_domain, &onnx_opset_version),
                          ort_api);
  size_t input_count;
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->OperatorGetNumInputs(op_type, onnx_opset_version, op_domain, &input_count),
                           ort_api);

  size_t output_count;
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->OperatorGetNumOutputs(op_type, onnx_opset_version, op_domain, &output_count),
                           ort_api);

  std::vector<const char*> input_names(input_count);
  for (size_t i = 0; i < input_count; i++) {
    const char* name;
    RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->OperatorGetInputName(op_type, onnx_opset_version, op_domain, i, &name),
                            ort_api);

    const char* actual_name;
    if (S_OK == GetValue(name, op_input_names, actual_input_names, num_inputs, &actual_name))
    {
      input_names[i] = actual_name;
    }
  }

  std::vector<const char*> output_names(output_count);
  for (size_t i = 0; i < output_count; i++) {
    const char* name;
    RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->OperatorGetOutputName(op_type, onnx_opset_version, op_domain, i, &name),
                             ort_api);
    const char* actual_name = nullptr;
    if (S_OK == GetValue(name, op_output_names, actual_output_names, num_outputs, &actual_name)) {
        output_names[i] = actual_name;
    }
  }

  std::vector<OrtValue*> attributes;
  for (size_t i = 0; i < num_attributes; i++) {
    attributes.push_back(static_cast<OnnxruntimeValue*>(*(attribute_values + i))->UseOrtValue());
  }

  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelAddOperator(
    ort_model_.get(), op_type, op_name, onnx_opset_version, op_domain, input_names.data(), input_count, output_names.data(), output_count, op_attribute_names, attributes.data(), num_attributes),
                          engine_factory_->UseOrtApi());
  return S_OK;
}

static ONNXTensorElementDataType
ONNXTensorElementDataTypeFromTensorKind(winml::TensorKind kind) {
  switch (kind) {
    case winml::TensorKind::Boolean: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
    case winml::TensorKind::String: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    }
    case winml::TensorKind::Float16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }
    case winml::TensorKind::Float: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    case winml::TensorKind::Double: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    }
    case winml::TensorKind::Int8: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    }
    case winml::TensorKind::Int16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    }
    case winml::TensorKind::Int32: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    }
    case winml::TensorKind::Int64: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    case winml::TensorKind::UInt8: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    }
    case winml::TensorKind::UInt16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    }
    case winml::TensorKind::UInt32: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    }
    case winml::TensorKind::UInt64: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    }
    case winml::TensorKind::Complex64: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
    }
    case winml::TensorKind::Complex128: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
    }
    default: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }
}

STDMETHODIMP OnnruntimeModel::AddModelInput(_In_ const char* const name, _In_ IDescriptorInfoProvider* descriptor_provider, bool is_constant, IValue* constant_value) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  auto ort_api = engine_factory_->UseOrtApi();

  winrt::com_ptr<_winml::IDescriptorInfo> descriptor_info;
  descriptor_provider->GetDescriptorInfo(engine_factory_.Get(), descriptor_info.put());

  auto ort_type_info_provider = descriptor_info.as<_winml::IOrtTypeInfoProvider>();
  OrtTypeInfo* type_info;
  ort_type_info_provider->GetTypeInfo(&type_info);

  if (is_constant) {
    RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelAddConstantInput(ort_model_.get(), name, type_info, static_cast<OnnxruntimeValue*>(constant_value)->UseOrtValue()),
                            ort_api);
  } else {
    RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelAddInput(ort_model_.get(), name, type_info),
                            ort_api);
  }

  return S_OK;
}

STDMETHODIMP OnnruntimeModel::AddModelOutput(_In_ const char* const name, _In_ IDescriptorInfoProvider* descriptor_provider) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  auto ort_api = engine_factory_->UseOrtApi();

  winrt::com_ptr<_winml::IDescriptorInfo> descriptor_info;
  descriptor_provider->GetDescriptorInfo(engine_factory_.Get(), descriptor_info.put());

  auto ort_type_info_provider = descriptor_info.as<_winml::IOrtTypeInfoProvider>();
  OrtTypeInfo* type_info;
  ort_type_info_provider->GetTypeInfo(&type_info);

  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelAddOutput(ort_model_.get(), name, type_info), ort_api);
  return S_OK;
}