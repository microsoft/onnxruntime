// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"
#include "OnnxruntimeModel.h"
#include "core/platform/windows/TraceLoggingConfig.h"
#include <evntrace.h>

#include "OnnxruntimeDescriptorConverter.h"
#include "OnnxruntimeEngine.h"
#include "OnnxruntimeErrors.h"

using namespace Windows::AI::MachineLearning;

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

HRESULT ModelInfo::RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory, OrtModel* ort_model) {
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

  WinML::OnnxruntimeDescriptorConverter converter(engine_factory, model_metadata_);

  static const winml_adapter_api_model_feature_helper input_helpers = {
      winml_adapter_api->ModelGetInputCount,
      winml_adapter_api->ModelGetInputName,
      winml_adapter_api->ModelGetInputDescription,
      winml_adapter_api->ModelGetInputTypeInfo};

  // Create inputs
  std::vector<OnnxruntimeValueInfoWrapper> inputs;
  RETURN_IF_FAILED(CreateFeatureDescriptors(engine_factory, &input_helpers, ort_model, inputs));
  input_features_ = converter.ConvertToLearningModelDescriptors(inputs);

  // Create outputs
  static const winml_adapter_api_model_feature_helper output_helpers = {
      winml_adapter_api->ModelGetOutputCount,
      winml_adapter_api->ModelGetOutputName,
      winml_adapter_api->ModelGetOutputDescription,
      winml_adapter_api->ModelGetOutputTypeInfo};

  std::vector<OnnxruntimeValueInfoWrapper> outputs;
  RETURN_IF_FAILED(CreateFeatureDescriptors(engine_factory, &output_helpers, ort_model, outputs));
  output_features_ = converter.ConvertToLearningModelDescriptors(outputs);

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

STDMETHODIMP ModelInfo::GetModelMetadata(ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING>** metadata) {
  std::unordered_map<winrt::hstring, winrt::hstring> map_copy;
  for (auto& pair : model_metadata_) {
    auto metadata_key = WinML::Strings::HStringFromUTF8(pair.first);
    auto metadata_value = WinML::Strings::HStringFromUTF8(pair.second);
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

STDMETHODIMP OnnruntimeModel::DetachOrtModel(OrtModel** model) {
  *model = ort_model_.release();
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::AddOperator(const char* const op_type, const char* const op_name, const char* const* input_names,
                                          size_t num_inputs, const char* const* output_names, size_t num_outputs) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelAddOperator(ort_model_.get(), op_type, op_name, input_names, num_inputs, output_names, num_outputs, nullptr, nullptr, 0),
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

STDMETHODIMP OnnruntimeModel::AddModelInput(_In_ const char* const name, _In_ IDescriptorInfoProvider* descriptor_provider, bool is_constant) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  auto ort_api = engine_factory_->UseOrtApi();

  winrt::com_ptr<WinML::IDescriptorInfo> descriptor_info;
  descriptor_provider->GetDescriptorInfo(engine_factory_.Get(), descriptor_info.put());

  auto ort_type_info_provider = descriptor_info.as<WinML::IOrtTypeInfoProvider>();
  OrtTypeInfo* type_info;
  ort_type_info_provider->GetTypeInfo(&type_info);

  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelAddInput(ort_model_.get(), name, type_info, is_constant),
                          ort_api);
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::AddModelOutput(_In_ const char* const name, _In_ IDescriptorInfoProvider* descriptor_provider) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  auto ort_api = engine_factory_->UseOrtApi();

  winrt::com_ptr<WinML::IDescriptorInfo> descriptor_info;
  descriptor_provider->GetDescriptorInfo(engine_factory_.Get(), descriptor_info.put());

  auto ort_type_info_provider = descriptor_info.as<WinML::IOrtTypeInfoProvider>();
  OrtTypeInfo* type_info;
  ort_type_info_provider->GetTypeInfo(&type_info);

  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ModelAddOutput(ort_model_.get(), name, type_info), ort_api);
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::InferOperatorOutputs(_In_ const char* const op_name, _In_ const wfc::IVector<winml::ILearningModelFeatureDescriptor>& inputs, _Out_ wfc::IVector<winml::ILearningModelFeatureDescriptor>& outputs) {
  UNREFERENCED_PARAMETER(op_name);
  UNREFERENCED_PARAMETER(inputs);
  UNREFERENCED_PARAMETER(outputs);
  return S_OK;
}

STDMETHODIMP OnnruntimeModel::ResolveOperatorInputs(_In_ const char* const op_type,
                                                    _In_ wfc::IVectorView<winml::ILearningModelFeatureDescriptor>& available_inputs,
                                                    _Out_ wfc::IVector<winml::ILearningModelFeatureDescriptor>& resolved_inputs,
                                                    _Out_ wfc::IMap<winrt::hstring, winrt::hstring>& mapping) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  std::vector<OrtTypeInfo*> available_inputs_vector;
  for (uint32_t i = 0; i < available_inputs.Size(); i++) {
    auto learning_model_descriptor = available_inputs.GetAt(i);
    auto descriptor_provider = learning_model_descriptor.as<IDescriptorInfoProvider>();

    winrt::com_ptr<IDescriptorInfo> descriptor_info;
    descriptor_provider->GetDescriptorInfo(engine_factory_.Get(), descriptor_info.put());
    auto ort_type_info_provider = descriptor_info.as<IOrtTypeInfoProvider>();

    OrtTypeInfo* info;
    ort_type_info_provider->GetTypeInfo(&info);

    available_inputs_vector.emplace_back(info);
  }

  size_t num_inputs;
  winml_adapter_api->OperatorGetNumInputs(op_type, &num_inputs);
  std::vector<size_t> indexes(num_inputs);
  winml_adapter_api->ResolveOperatorInputs(op_type, available_inputs_vector.data(), available_inputs_vector.size(), indexes.data(), num_inputs);

  resolved_inputs.Clear();
  mapping.Clear();
  for (size_t i = 0; i < num_inputs; i++) {
    auto feature_descriptor = available_inputs.GetAt(static_cast<uint32_t>(indexes[i]));
    resolved_inputs.Append(feature_descriptor);

    const char* name;
    winml_adapter_api->OperatorGetInputName(op_type, i, &name);
    mapping.Insert(WinML::Strings::HStringFromUTF8(name), feature_descriptor.Name());
  }

  return S_OK;
}