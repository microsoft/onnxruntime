// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"
#include "ConverterResourceStore.h"
#include "impl/FeatureCompatibility.h"
#include "FeatureValues.h"
#include "LearningModelBinding.h"
#include "LearningModelSession.h"
#include "TelemetryEvent.h"
#include <onnxruntime_c_api.h>
#include "LearningModel.h"

using namespace WinML;

namespace winrt::Windows::AI::MachineLearning::implementation {
LearningModelBinding::LearningModelBinding(
    Windows::AI::MachineLearning::LearningModelSession const& session) try : m_session(session) {
  session.as<winmlp::LearningModelSession>()->CheckClosed();
}
WINML_CATCH_ALL

static Windows::AI::MachineLearning::ILearningModelFeatureDescriptor FindValidBinding(
    winrt::Windows::Foundation::Collections::IIterable<ILearningModelFeatureDescriptor> descriptors,
    const std::wstring& name) {
  for (auto descriptor : descriptors) {
    auto descriptor_native = descriptor.as<ILearningModelFeatureDescriptorNative>();

    const wchar_t* feature_name;
    uint32_t size;
    WINML_THROW_IF_FAILED(descriptor_native->GetName(&feature_name, &size));

    // Case insensetive comparison of onnx name in feature descriptor, and passed in name
    if (_wcsicmp(feature_name, name.c_str()) == 0) {
      return descriptor;
    }
  }
  return nullptr;
}

using NullableBindingPort = std::optional<std::pair<Windows::AI::MachineLearning::ILearningModelFeatureDescriptor, BindingType>>;

static NullableBindingPort FindValidBinding(
    winml::LearningModel model,
    const std::wstring& name) {
  if (auto descriptor = FindValidBinding(model.InputFeatures(), name)) {
    return std::make_pair(descriptor, BindingType::kInput);
  } else if (auto output_descriptor = FindValidBinding(model.OutputFeatures(), name)) {
    return std::make_pair(output_descriptor, BindingType::kOutput);
  }

  return {};
}

void LearningModelBinding::CacheProvider(
    std::string name,
    ProviderInfo& providerInfo) {
  m_providers[name] = providerInfo;
}

std::tuple<std::string, winrt::com_ptr<WinML::IValue>, BindingType> LearningModelBinding::CreateBinding(
    const std::string& name,
    const Windows::Foundation::IInspectable& inspectable,
    Windows::Foundation::Collections::IPropertySet const& properties) {
  // Given a known type, validate against the model
  auto model = m_session.Model();
  auto bindingPort = FindValidBinding(model, WinML::Strings::WStringFromString(name));

  WINML_THROW_HR_IF_FALSE_MSG(
      WINML_ERR_INVALID_BINDING,
      bindingPort.has_value(),
      "The model has no variable with name %s.",
      name.c_str());

  // Retrieve the descriptor and binding type
  auto descriptor = bindingPort->first;
  auto bindingType = bindingPort->second;

  // Create a feature value from the iinspectable input
  auto featureValue = WinML::CreateFeatureValueFromInspectable(bindingType, inspectable, descriptor);
  WINML_THROW_HR_IF_NULL_MSG(
      WINML_ERR_INVALID_BINDING,
      featureValue,
      "The model variable %s cannot be bound with the provided type.",
      name.c_str());

  // Validate that the feature value is compatible with the descriptor
  WinML::VerifyFeatureValueCompatibleWithDescriptor(featureValue, descriptor);

  // Create the Binding Context to pass to the feature value
  BindingContext context{
      bindingType,
      m_session,
      descriptor,
      properties,
      {}  // SubresourceId is set by callee
  };

  // Get the bound tensor
  winrt::com_ptr<IValue> value;

  // Get the native interface for the given bind value
  auto spLotusValueProvider = featureValue.as<WinML::ILotusValueProviderPrivate>();

  auto spSession = m_session.as<LearningModelSession>();

  // Check if the feature value is a placeholder
  bool isPlaceHolder;
  WINML_THROW_IF_FAILED(spLotusValueProvider->IsPlaceholder(&isPlaceHolder));

  // If binding a tensor for gpu execution, always bind.
  // If it is a placeholder, gpu resources will be preallocated during bind.
  // This enables the chaining scenario.
  auto spDevice = m_session.Device().as<LearningModelDevice>();
  auto isGpuSession = !spDevice->IsCpuDevice();
  auto spTensor = featureValue.try_as<ITensor>();
  auto isTensorWithShape = spTensor != nullptr && spTensor.Shape().Size() != 0;
  auto shouldAlwaysTensorize = isTensorWithShape && isGpuSession;

  if (!isPlaceHolder || shouldAlwaysTensorize) {
    // If not a placeholder, attempt to get the underlying resource
    WINML_THROW_IF_FAILED_MSG(
        spLotusValueProvider->GetValue(context, value.put()),
        "The model variable %s failed tensorization.",
        name.c_str());
  } else {
    WINML_THROW_HR_IF_TRUE_MSG(
        WINML_ERR_INVALID_BINDING,
        isPlaceHolder && bindingType == BindingType::kInput,
        "The model variable %s is an input, but has no associated resources to bind.",
        name.c_str());

    WINML_THROW_IF_FAILED(spSession->GetEngine()->CreateNullValue(value.put()));
  }

  // Hold onto the input output providers so that our memory doesnt get destroyed!
  auto providerInfo = ProviderInfo{inspectable, spLotusValueProvider, context};
  CacheProvider(name, providerInfo);
  
  return std::make_tuple(name, value, bindingType);
}

void LearningModelBinding::Bind(
    hstring const& name,
    Windows::Foundation::IInspectable const& value) try {
  return Bind(name, value, nullptr /* no properties */);
}
WINML_CATCH_ALL

void LearningModelBinding::Bind(
    hstring const& name,
    Windows::Foundation::IInspectable const& value,
    Windows::Foundation::Collections::IPropertySet const& properties) try {
  _winmlt::TelemetryEvent binding_event(_winmlt::EventCategory::kBinding);

  BindingType binding_type;
  std::string binding_name;
  winrt::com_ptr<WinML::IValue> binding_value = nullptr;
  auto featureName = WinML::Strings::UTF8FromHString(name);
  std::tie(binding_name, binding_value, binding_type) = CreateBinding(featureName, value, properties);
  switch (binding_type) {
    case BindingType::kInput:
      WINML_THROW_IF_FAILED(BindInput(binding_name, binding_value));
      break;
    case BindingType::kOutput:
      WINML_THROW_IF_FAILED(BindOutput(binding_name, binding_value));
      break;
    default:
      FAIL_FAST();
  }
}
WINML_CATCH_ALL

void LearningModelBinding::Clear() try {
  m_session.as<winmlp::LearningModelSession>()->CheckClosed();
  inputs_.clear();
  input_names_.clear();
  outputs_.clear();
  output_names_.clear();
  m_providers.clear();
}
WINML_CATCH_ALL

Windows::Foundation::Collections::IIterator<LearningModelBinding::KeyValuePair> LearningModelBinding::First() {
  std::unordered_map<hstring, Windows::Foundation::IInspectable> bindingsMap;

  for (auto mergedBindings : m_providers) {
    auto name = WinML::Strings::HStringFromUTF8(mergedBindings.first);
    bindingsMap[name] = mergedBindings.second.CallerSpecifiedFeatureValue;
  }

  return winrt::single_threaded_map(std::move(bindingsMap)).First();
}

Windows::Foundation::IInspectable LearningModelBinding::Lookup(hstring const& key) {
  auto utf8_name = WinML::Strings::UTF8FromHString(key);

  auto foundIt = m_providers.find(utf8_name);
  WINML_THROW_HR_IF_FALSE_MSG(
      E_BOUNDS,
      foundIt != std::end(m_providers),
      "The binding collection does not contain a variable with name %s.",
      utf8_name.c_str());

  auto providerInfo = foundIt->second;
  return providerInfo.CallerSpecifiedFeatureValue;
}

uint32_t LearningModelBinding::Size() {
  return static_cast<uint32_t>(m_providers.size());
}

bool LearningModelBinding::HasKey(hstring const& key) {
  auto utf8_name = WinML::Strings::UTF8FromHString(key);
  return m_providers.find(utf8_name) != m_providers.end();
}

void LearningModelBinding::Split(
    Windows::Foundation::Collections::IMapView<hstring, Windows::Foundation::IInspectable>& first,
    Windows::Foundation::Collections::IMapView<hstring, Windows::Foundation::IInspectable>& second) {
  // the winrt api guide states:
  // If the IMapView instance cannot be split, then both the first and second parameters are null when the method returns.
  first = nullptr;
  second = nullptr;
}

ILearningModelFeatureValue LearningModelBinding::CreateUnboundOuputFeatureValue(
    const winrt::com_ptr<IValue> value,
    ILearningModelFeatureDescriptor& descriptor) {
  bool out;
  if (SUCCEEDED(value->IsTensor(&out)) && out) {
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Float, &out)) && out) {
      if (descriptor.Kind() == LearningModelFeatureKind::Image) {
        using namespace Windows::Graphics::Imaging;
        // TODO: this format for unbound output needs more discussion
        BitmapPixelFormat format = descriptor.as<ImageFeatureDescriptor>()->BitmapPixelFormat();
        std::vector<int64_t> shape;
        value->GetTensorShape(shape);
        uint32_t width = static_cast<uint32_t>(shape[3]);
        uint32_t height = static_cast<uint32_t>(shape[2]);
        uint32_t batchSize = static_cast<uint32_t>(shape[0]);
        return implementation::ImageFeatureValue::Create(batchSize, format, width, height);
      } else {
        return implementation::TensorFloat::Create();
      }
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Double, &out)) && out) {
      return implementation::TensorDouble::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::String, &out)) && out) {
      return implementation::TensorString::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::UInt8, &out)) && out) {
      return implementation::TensorUInt8Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Int8, &out)) && out) {
      return implementation::TensorInt8Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::UInt16, &out)) && out) {
      return implementation::TensorUInt16Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Int16, &out)) && out) {
      return implementation::TensorInt16Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::UInt32, &out)) && out) {
      return implementation::TensorUInt32Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Int32, &out)) && out) {
      return implementation::TensorInt32Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::UInt64, &out)) && out) {
      return implementation::TensorUInt64Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Int64, &out)) && out) {
      return implementation::TensorInt64Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Boolean, &out)) && out) {
      return implementation::TensorBoolean::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Float16, &out)) && out) {
      return implementation::TensorFloat16Bit::Create();
    }
  }

  // Maps
  if (SUCCEEDED(value->IsOfMapType(TensorKind::String, TensorKind::String, &out)) && out) {
    return implementation::MapStringToString::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::String, TensorKind::Int64, &out)) && out) {
    return implementation::MapStringToInt64Bit::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::String, TensorKind::Float, &out)) && out) {
    return implementation::MapStringToFloat::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::String, TensorKind::Double, &out)) && out) {
    return implementation::MapStringToDouble::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::Int64, TensorKind::String, &out)) && out) {
    return implementation::MapInt64BitToString::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::Int64, TensorKind::Int64, &out)) && out) {
    return implementation::MapInt64BitToInt64Bit::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::Int64, TensorKind::Float, &out)) && out) {
    return implementation::MapInt64BitToFloat::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::Int64, TensorKind::Double, &out)) && out) {
    return implementation::MapInt64BitToDouble::Create();
  }
  // Sequences
  if (SUCCEEDED(value->IsOfVectorMapType(TensorKind::String, TensorKind::Float, &out)) && out) {
    return implementation::SequenceMapStringFloat::Create();
  }
  if (SUCCEEDED(value->IsOfVectorMapType(TensorKind::Int64, TensorKind::Float, &out)) && out) {
    return implementation::SequenceMapInt64BitFloat::Create();
  }

  auto utf8_name = WinML::Strings::UTF8FromHString(descriptor.Name());
  WINML_THROW_HR_IF_TRUE_MSG(
      E_UNEXPECTED,
      true,
      "The engine produced an unexpected evaluation output for unbound output variable %s.",
      utf8_name.c_str());

  return nullptr;
}

Windows::Foundation::IInspectable LearningModelBinding::CreateUnboundOutput(
    const std::string& name,
    winrt::com_ptr<WinML::IValue> value) {
  // Find valid binding port
  auto bindingPort = FindValidBinding(
      m_session.Model(),
      WinML::Strings::WStringFromString(name));

  WINML_THROW_HR_IF_FALSE_MSG(
      E_UNEXPECTED,
      bindingPort.has_value(),
      "The engine produced an unexpected evaluation output %s, that is not a model variable.",
      name.c_str());

  // Retrieve the descriptor and binding type
  auto descriptor = bindingPort->first;
  auto bindingType = bindingPort->second;
  WINML_THROW_HR_IF_FALSE_MSG(
      E_UNEXPECTED,
      bindingType == BindingType::kOutput,
      "The engine produced an unexpected evaluation output %s, that is not a model variable output.",
      name.c_str());

  // Create a binding context
  BindingContext context{
      bindingType,
      m_session,
      descriptor,
      nullptr /* no binding properties for unbound outputs */,
      {}  // SubresourceId is set by callee
  };

  // Create empty feature value
  auto featureValue = CreateUnboundOuputFeatureValue(value, descriptor);

  // Update feature value
  auto spLotusValueProvider = featureValue.as<WinML::ILotusValueProviderPrivate>();
  WINML_THROW_IF_FAILED_MSG(
      spLotusValueProvider->UpdateSourceResourceData(context, value.get()),
      "Failed to update bound object for model variable output %s",
      name.c_str());

  // Get abi representation
  winrt::Windows::Foundation::IInspectable inspectable;
  WINML_THROW_IF_FAILED_MSG(
      spLotusValueProvider->AbiRepresentation(inspectable),
      "Failed to return bound object for model variable output %s",
      name.c_str());

  return inspectable;
}

std::unordered_map<std::string, Windows::Foundation::IInspectable> LearningModelBinding::UpdateProviders() {
  std::unordered_map<std::string, Windows::Foundation::IInspectable> outputs;

  auto& output_names = GetOutputNames();
  auto& output_values = GetOutputs();
  WINML_THROW_HR_IF_FALSE_MSG(
      E_UNEXPECTED,
      output_names.size() == output_values.size(),
      "Evaluation produced unexpected output variables.");

  for (unsigned i = 0; i < output_names.size(); i++) {
    auto utf8_name = output_names[i];
    auto value = output_values[i];

    if (m_providers.find(utf8_name) != std::end(m_providers)) {
      auto& providerInfo = m_providers[utf8_name];
      auto provider = providerInfo.Provider;
      auto context = providerInfo.Context;
      WINML_THROW_IF_FAILED_MSG(
          provider->UpdateSourceResourceData(context, value.get()),
          "Failed to update bound object for model variable output %s",
          utf8_name.c_str());

      outputs[utf8_name] = providerInfo.CallerSpecifiedFeatureValue;
    } else {
      // unbound outputs
      outputs[utf8_name] = CreateUnboundOutput(utf8_name, value);
    }
  }

  // Clear any converters cached on inputs to return them to the pool
  for (auto&& provider : m_providers) {
    if (provider.second.Context.converter != nullptr) {
      provider.second.Context.converter->Get()->Tensorizer->ResetAllocator();
      provider.second.Context.converter = nullptr;
    }
  }

  return outputs;
}

STDMETHODIMP LearningModelBinding::Bind(
    const wchar_t* name,
    UINT32 cchName,
    IUnknown* value) {
  try {
    _winmlt::TelemetryEvent binding_event(_winmlt::EventCategory::kBinding);
    BindingType binding_type;
    std::string binding_name;
    winrt::com_ptr<WinML::IValue> binding_value;

    winrt::Windows::Foundation::IInspectable to;
    RETURN_IF_FAILED(value->QueryInterface(
        winrt::guid_of<winrt::Windows::Foundation::IInspectable>(),
        reinterpret_cast<void**>(winrt::put_abi(to))));

    auto featureName = WinML::Strings::UTF8FromUnicode(name, cchName);
    std::tie(binding_name, binding_value, binding_type) = CreateBinding(featureName, to, nullptr);
    switch (binding_type) {
      case BindingType::kInput:
        WINML_THROW_IF_FAILED(BindInput(binding_name, binding_value));
        break;
      case BindingType::kOutput:
        WINML_THROW_IF_FAILED(BindOutput(binding_name, binding_value));
        break;
      default:
        FAIL_FAST();
    }
    return S_OK;
  }
  WINML_CATCH_ALL_COM
}

static std::pair<bool, size_t> Contains(const std::vector<std::string>& names, const std::string& name) {
  auto it = std::find(std::begin(names), std::end(names), name);
  if (it == std::end(names)) {
    return {false, 0};
  }
  return {true, it - std::begin(names)};
}

// This method releases control of memory of ml_value from caller of BindInput
HRESULT LearningModelBinding::BindInput(const std::string& name, winrt::com_ptr<WinML::IValue> value) {
  bool exists;
  size_t index;
  std::tie(exists, index) = Contains(input_names_, name);

  auto engine = m_session.as<LearningModelSession>()->GetEngine();
  winrt::com_ptr<WinML::IValue> device_value;
  WINML_THROW_IF_FAILED(engine->CreateOneInputAcrossDevices(name.c_str(), value.get(), device_value.put()));  // an input will always be copied on device mismatch

  if (exists) {
    inputs_[index] = device_value;
  } else {
    input_names_.push_back(name);
    inputs_.push_back(device_value);
  }

  return S_OK;
}

HRESULT LearningModelBinding::BindOutput(const std::string& name, winrt::com_ptr<WinML::IValue> value) {
  bool exists;
  size_t index;
  std::tie(exists, index) = Contains(output_names_, name);

  if (exists) {
    outputs_[index] = value;
    return S_OK;
  }

  output_names_.push_back(name);
  outputs_.push_back(value);
  return S_OK;
}

const std::vector<std::string>& LearningModelBinding::GetOutputNames() const {
  return output_names_;
}

const std::vector<std::string>& LearningModelBinding::GetInputNames() const {
  return input_names_;
}

std::vector<winrt::com_ptr<WinML::IValue>>& LearningModelBinding::GetOutputs() {
  return outputs_;
}

const std::vector<winrt::com_ptr<WinML::IValue>>& LearningModelBinding::GetInputs() const {
  return inputs_;
}

void LearningModelBinding::BindUnboundOutputs() {
  auto& bound_output_names = GetOutputNames();
  std::unordered_set<std::string> bound_output_names_set(
      bound_output_names.begin(),
      bound_output_names.end());

  // Get model output feature names
  auto model_impl = m_session.Model().as<winmlp::LearningModel>();
  auto output_features = model_impl->OutputFeatures();
  std::vector<ILearningModelFeatureDescriptor> output_descriptors(
      begin(output_features),
      end(output_features));

  // Convert all output features to their feature names
  std::vector<std::string> output_feature_names;
  std::transform(
      std::begin(output_descriptors),
      std::end(output_descriptors),
      std::back_inserter(output_feature_names),
      [&](auto& descriptor) {
        auto descriptor_native = descriptor.as<ILearningModelFeatureDescriptorNative>();
        const wchar_t* p_name;
        uint32_t size;
        WINML_THROW_IF_FAILED(descriptor_native->GetName(&p_name, &size));
        return WinML::Strings::UTF8FromUnicode(p_name, size);
      });

  // Find the set difference to determine if there are any unbound output features
  std::vector<std::string> unbound_output_names;
  std::copy_if(
      std::begin(output_feature_names), std::end(output_feature_names),
      std::inserter(unbound_output_names, std::begin(unbound_output_names)),
      [&](const auto& outputFeatureName) {
        return bound_output_names_set.find(outputFeatureName) == bound_output_names_set.end();
      });

  // Add all unbound outputs to binding collection
  for (const auto& unbound_output : unbound_output_names) {
    auto engine = m_session.as<LearningModelSession>()->GetEngine();

    winrt::com_ptr<IValue> value;
    WINML_THROW_IF_FAILED(engine->CreateNullValue(value.put()));
    WINML_THROW_IF_FAILED(BindOutput(unbound_output, value));
  }
}

}  // namespace winrt::Windows::AI::MachineLearning::implementation