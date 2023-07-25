// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api/pch/pch.h"
#include "ConverterResourceStore.h"
#include "impl/FeatureCompatibility.h"
#include "FeatureValues.h"
#include "LearningModelBinding.h"
#include "LearningModelSession.h"
#include "TelemetryEvent.h"
#include "LearningModel.h"

namespace WINMLP {
LearningModelBinding::~LearningModelBinding() {
  Clear();
}

LearningModelBinding::LearningModelBinding(winml::LearningModelSession const& session) try : m_session(session) {
  session.as<winmlp::LearningModelSession>()->CheckClosed();
}
WINML_CATCH_ALL

static winml::ILearningModelFeatureDescriptor FindValidBinding(
  wfc::IIterable<ILearningModelFeatureDescriptor> descriptors, const std::wstring& name
) {
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

using NullableBindingPort = std::optional<std::pair<winml::ILearningModelFeatureDescriptor, _winml::BindingType>>;

static NullableBindingPort FindValidBinding(winml::LearningModel model, const std::wstring& name) {
  if (auto descriptor = FindValidBinding(model.InputFeatures(), name)) {
    return std::make_pair(descriptor, _winml::BindingType::kInput);
  } else if (auto output_descriptor = FindValidBinding(model.OutputFeatures(), name)) {
    return std::make_pair(output_descriptor, _winml::BindingType::kOutput);
  }

  return {};
}

void LearningModelBinding::CacheProvider(std::string name, ProviderInfo& providerInfo) {
  m_providers[name] = providerInfo;
}

std::tuple<std::string, winrt::com_ptr<_winml::IValue>, _winml::BindingType> LearningModelBinding::CreateBinding(
  const std::string& name, const wf::IInspectable& inspectable, wfc::IPropertySet const& properties
) {
  // Given a known type, validate against the model
  auto model = m_session.Model();
  auto bindingPort = FindValidBinding(model, _winml::Strings::WStringFromString(name));

  WINML_THROW_HR_IF_FALSE_MSG(
    WINML_ERR_INVALID_BINDING, bindingPort.has_value(), "The model has no variable with name %s.", name.c_str()
  );

  // Retrieve the descriptor and binding type
  auto descriptor = bindingPort->first;
  auto bindingType = bindingPort->second;

  // Create a feature value from the iinspectable input
  auto featureValue = _winml::CreateFeatureValueFromInspectable(bindingType, inspectable, descriptor);
  WINML_THROW_HR_IF_NULL_MSG(
    WINML_ERR_INVALID_BINDING,
    featureValue,
    "The model variable %s cannot be bound with the provided type.",
    name.c_str()
  );

  // Validate that the feature value is compatible with the descriptor
  _winml::VerifyFeatureValueCompatibleWithDescriptor(featureValue, descriptor);

  // Create the Binding Context to pass to the feature value
  _winml::BindingContext context{
    bindingType, m_session, descriptor, properties, {}  // SubresourceId is set by callee
  };

  // Get the bound tensor
  winrt::com_ptr<_winml::IValue> value;

  // Get the native interface for the given bind value
  auto spLotusValueProvider = featureValue.as<_winml::ILotusValueProviderPrivate>();

  auto spSession = m_session.as<LearningModelSession>();

  // Check if the feature value is a placeholder
  bool isPlaceHolder;
  WINML_THROW_IF_FAILED(spLotusValueProvider->IsPlaceholder(&isPlaceHolder));

  // If binding a tensor for gpu execution, always bind.
  // If it is a placeholder, gpu resources will be preallocated during bind.
  // This enables the chaining scenario.
  auto spDevice = m_session.Device().as<LearningModelDevice>();
  auto isGpuSession = !spDevice->IsCpuDevice();
  auto spTensor = featureValue.try_as<winml::ITensor>();
  auto isTensorWithShape = spTensor != nullptr && spTensor.Shape().Size() != 0;
  auto shouldAlwaysTensorize = isTensorWithShape && isGpuSession;

  if (!isPlaceHolder || shouldAlwaysTensorize) {
    // If not a placeholder, attempt to get the underlying resource
    WINML_THROW_IF_FAILED_MSG(
      spLotusValueProvider->GetValue(context, value.put()), "The model variable %s failed tensorization.", name.c_str()
    );
  } else {
    WINML_THROW_HR_IF_TRUE_MSG(
      WINML_ERR_INVALID_BINDING,
      isPlaceHolder && bindingType == _winml::BindingType::kInput,
      "The model variable %s is an input, but has no associated resources to bind.",
      name.c_str()
    );

    WINML_THROW_IF_FAILED(spSession->GetEngine()->CreateNullValue(value.put()));
  }

  // Hold onto the input output providers so that our memory doesnt get destroyed!
  auto providerInfo = ProviderInfo{inspectable, spLotusValueProvider, context};
  CacheProvider(name, providerInfo);

  return std::make_tuple(name, value, bindingType);
}

void LearningModelBinding::Bind(hstring const& name, wf::IInspectable const& value) try {
  return Bind(name, value, nullptr /* no properties */);
}
WINML_CATCH_ALL

void LearningModelBinding::Bind(
  hstring const& name, wf::IInspectable const& value, wfc::IPropertySet const& properties
) try {
  // if this is being called on the GPU, grab the DML lock
  // the DML EP is not thread safe.
  auto session = m_session.as<winmlp::LearningModelSession>();
  auto device = m_session.Device().as<winmlp::LearningModelDevice>();
  CWinMLAutoLock lock(!device->IsCpuDevice() ? session->GetDMLEPLock() : nullptr);

  _winmlt::TelemetryEvent binding_event(_winmlt::EventCategory::kBinding);

  _winml::BindingType binding_type;
  std::string binding_name;
  winrt::com_ptr<_winml::IValue> binding_value = nullptr;
  auto featureName = _winml::Strings::UTF8FromHString(name);
  std::tie(binding_name, binding_value, binding_type) = CreateBinding(featureName, value, properties);
  switch (binding_type) {
    case _winml::BindingType::kInput:
      WINML_THROW_IF_FAILED(BindInput(binding_name, binding_value));
      break;
    case _winml::BindingType::kOutput:
      WINML_THROW_IF_FAILED(BindOutput(binding_name, binding_value));
      break;
    default:
      FAIL_FAST();
  }
}
WINML_CATCH_ALL

void LearningModelBinding::Clear() try {
  // if this is being called on the GPU, grab the DML lock
  // the DML EP is not thread safe.
  auto session = m_session.as<winmlp::LearningModelSession>();
  auto device = m_session.Device().as<winmlp::LearningModelDevice>();
  CWinMLAutoLock lock(!device->IsCpuDevice() ? session->GetDMLEPLock() : nullptr);

  inputs_.clear();
  input_names_.clear();
  outputs_.clear();
  output_names_.clear();
  m_providers.clear();
}
WINML_CATCH_ALL

wfc::IIterator<LearningModelBinding::KeyValuePair> LearningModelBinding::First() {
  std::unordered_map<hstring, wf::IInspectable> bindingsMap;

  for (auto mergedBindings : m_providers) {
    auto name = _winml::Strings::HStringFromUTF8(mergedBindings.first);
    bindingsMap[name] = mergedBindings.second.CallerSpecifiedFeatureValue;
  }

  return winrt::single_threaded_map(std::move(bindingsMap)).First();
}

wf::IInspectable LearningModelBinding::Lookup(hstring const& key) {
  auto utf8_name = _winml::Strings::UTF8FromHString(key);

  auto foundIt = m_providers.find(utf8_name);
  WINML_THROW_HR_IF_FALSE_MSG(
    E_BOUNDS,
    foundIt != std::end(m_providers),
    "The binding collection does not contain a variable with name %s.",
    utf8_name.c_str()
  );

  auto providerInfo = foundIt->second;
  return providerInfo.CallerSpecifiedFeatureValue;
}

uint32_t LearningModelBinding::Size() {
  return static_cast<uint32_t>(m_providers.size());
}

bool LearningModelBinding::HasKey(hstring const& key) {
  auto utf8_name = _winml::Strings::UTF8FromHString(key);
  return m_providers.find(utf8_name) != m_providers.end();
}

void LearningModelBinding::Split(
  wfc::IMapView<hstring, wf::IInspectable>& first, wfc::IMapView<hstring, wf::IInspectable>& second
) {
  // the winrt api guide states:
  // If the IMapView instance cannot be split, then both the first and second parameters are null when the method returns.
  first = nullptr;
  second = nullptr;
}

ILearningModelFeatureValue LearningModelBinding::CreateUnboundOuputFeatureValue(
  const winrt::com_ptr<_winml::IValue> value, ILearningModelFeatureDescriptor& descriptor
) {
  bool out;
  if (SUCCEEDED(value->IsTensor(&out)) && out) {
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Float, &out)) && out) {
      if (descriptor.Kind() == LearningModelFeatureKind::Image) {
        // TODO: this format for unbound output needs more discussion
        wgi::BitmapPixelFormat format = descriptor.as<ImageFeatureDescriptor>()->BitmapPixelFormat();
        std::vector<int64_t> shape;
        value->GetTensorShape(shape);
        uint32_t width = static_cast<uint32_t>(shape[3]);
        uint32_t height = static_cast<uint32_t>(shape[2]);
        uint32_t batchSize = static_cast<uint32_t>(shape[0]);
        return winmlp::ImageFeatureValue::Create(batchSize, format, width, height);
      } else {
        return winmlp::TensorFloat::Create();
      }
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Double, &out)) && out) {
      return winmlp::TensorDouble::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::String, &out)) && out) {
      return winmlp::TensorString::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::UInt8, &out)) && out) {
      return winmlp::TensorUInt8Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Int8, &out)) && out) {
      return winmlp::TensorInt8Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::UInt16, &out)) && out) {
      return winmlp::TensorUInt16Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Int16, &out)) && out) {
      return winmlp::TensorInt16Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::UInt32, &out)) && out) {
      return winmlp::TensorUInt32Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Int32, &out)) && out) {
      return winmlp::TensorInt32Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::UInt64, &out)) && out) {
      return winmlp::TensorUInt64Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Int64, &out)) && out) {
      return winmlp::TensorInt64Bit::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Boolean, &out)) && out) {
      return winmlp::TensorBoolean::Create();
    }
    if (SUCCEEDED(value->IsOfTensorType(TensorKind::Float16, &out)) && out) {
      return winmlp::TensorFloat16Bit::Create();
    }
  }

  // Maps
  if (SUCCEEDED(value->IsOfMapType(TensorKind::String, TensorKind::String, &out)) && out) {
    return winmlp::MapStringToString::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::String, TensorKind::Int64, &out)) && out) {
    return winmlp::MapStringToInt64Bit::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::String, TensorKind::Float, &out)) && out) {
    return winmlp::MapStringToFloat::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::String, TensorKind::Double, &out)) && out) {
    return winmlp::MapStringToDouble::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::Int64, TensorKind::String, &out)) && out) {
    return winmlp::MapInt64BitToString::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::Int64, TensorKind::Int64, &out)) && out) {
    return winmlp::MapInt64BitToInt64Bit::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::Int64, TensorKind::Float, &out)) && out) {
    return winmlp::MapInt64BitToFloat::Create();
  }
  if (SUCCEEDED(value->IsOfMapType(TensorKind::Int64, TensorKind::Double, &out)) && out) {
    return winmlp::MapInt64BitToDouble::Create();
  }
  // Sequences
  if (SUCCEEDED(value->IsOfVectorMapType(TensorKind::String, TensorKind::Float, &out)) && out) {
    return winmlp::SequenceMapStringFloat::Create();
  }
  if (SUCCEEDED(value->IsOfVectorMapType(TensorKind::Int64, TensorKind::Float, &out)) && out) {
    return winmlp::SequenceMapInt64BitFloat::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::Float, &out)) && out) {
    return winmlp::SequenceTensorFloat::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::Double, &out)) && out) {
    return winmlp::SequenceTensorDouble::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::String, &out)) && out) {
    return winmlp::SequenceTensorString::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::UInt8, &out)) && out) {
    return winmlp::SequenceTensorUInt8Bit::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::Int8, &out)) && out) {
    return winmlp::SequenceTensorInt8Bit::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::UInt16, &out)) && out) {
    return winmlp::SequenceTensorUInt16Bit::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::Int16, &out)) && out) {
    return winmlp::SequenceTensorInt16Bit::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::UInt32, &out)) && out) {
    return winmlp::SequenceTensorUInt32Bit::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::Int32, &out)) && out) {
    return winmlp::SequenceTensorInt32Bit::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::UInt64, &out)) && out) {
    return winmlp::SequenceTensorUInt64Bit::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::Int64, &out)) && out) {
    return winmlp::SequenceTensorInt64Bit::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::Boolean, &out)) && out) {
    return winmlp::SequenceTensorBoolean::Create();
  }
  if (SUCCEEDED(value->IsOfVectorTensorType(TensorKind::Float16, &out)) && out) {
    return winmlp::SequenceTensorFloat16Bit::Create();
  }

  auto utf8_name = _winml::Strings::UTF8FromHString(descriptor.Name());
  WINML_THROW_HR_IF_TRUE_MSG(
    E_UNEXPECTED,
    true,
    "The engine produced an unexpected evaluation output for unbound output variable %s.",
    utf8_name.c_str()
  );

  return nullptr;
}

wf::IInspectable LearningModelBinding::CreateUnboundOutput(
  const std::string& name, winrt::com_ptr<_winml::IValue> value
) {
  // Find valid binding port
  auto bindingPort = FindValidBinding(m_session.Model(), _winml::Strings::WStringFromString(name));

  WINML_THROW_HR_IF_FALSE_MSG(
    E_UNEXPECTED,
    bindingPort.has_value(),
    "The engine produced an unexpected evaluation output %s, that is not a model variable.",
    name.c_str()
  );

  // Retrieve the descriptor and binding type
  auto descriptor = bindingPort->first;
  auto bindingType = bindingPort->second;
  WINML_THROW_HR_IF_FALSE_MSG(
    E_UNEXPECTED,
    bindingType == _winml::BindingType::kOutput,
    "The engine produced an unexpected evaluation output %s, that is not a model variable output.",
    name.c_str()
  );

  // Create a binding context
  _winml::BindingContext context{
    bindingType,
    m_session,
    descriptor,
    nullptr /* no binding properties for unbound outputs */,
    {}  // SubresourceId is set by callee
  };

  // Create empty feature value
  auto featureValue = CreateUnboundOuputFeatureValue(value, descriptor);

  // Update feature value
  auto spLotusValueProvider = featureValue.as<_winml::ILotusValueProviderPrivate>();
  WINML_THROW_IF_FAILED_MSG(
    spLotusValueProvider->UpdateSourceResourceData(context, value.get()),
    "Failed to update bound object for model variable output %s",
    name.c_str()
  );

  // Get abi representation
  wf::IInspectable inspectable;
  WINML_THROW_IF_FAILED_MSG(
    spLotusValueProvider->AbiRepresentation(inspectable),
    "Failed to return bound object for model variable output %s",
    name.c_str()
  );

  return inspectable;
}

std::unordered_map<std::string, wf::IInspectable> LearningModelBinding::UpdateProviders() {
  std::unordered_map<std::string, wf::IInspectable> outputs;

  auto& output_names = GetOutputNames();
  auto& output_values = GetOutputs();
  WINML_THROW_HR_IF_FALSE_MSG(
    E_UNEXPECTED, output_names.size() == output_values.size(), "Evaluation produced unexpected output variables."
  );

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
        utf8_name.c_str()
      );

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

STDMETHODIMP LearningModelBinding::Bind(const wchar_t* name, UINT32 cchName, IUnknown* value) {
  try {
    // if this is being called on the GPU, grab the DML lock
    // the DML EP is not thread safe.
    auto session = m_session.as<winmlp::LearningModelSession>();
    auto device = m_session.Device().as<winmlp::LearningModelDevice>();
    CWinMLAutoLock lock(!device->IsCpuDevice() ? session->GetDMLEPLock() : nullptr);

    _winmlt::TelemetryEvent binding_event(_winmlt::EventCategory::kBinding);
    _winml::BindingType binding_type;
    std::string binding_name;
    winrt::com_ptr<_winml::IValue> binding_value;

    wf::IInspectable to;
    RETURN_IF_FAILED(
      value->QueryInterface(winrt::guid_of<wf::IInspectable>(), reinterpret_cast<void**>(winrt::put_abi(to)))
    );

    auto featureName = _winml::Strings::UTF8FromUnicode(name, cchName);
    std::tie(binding_name, binding_value, binding_type) = CreateBinding(featureName, to, nullptr);
    switch (binding_type) {
      case _winml::BindingType::kInput:
        WINML_THROW_IF_FAILED(BindInput(binding_name, binding_value));
        break;
      case _winml::BindingType::kOutput:
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
HRESULT LearningModelBinding::BindInput(const std::string& name, winrt::com_ptr<_winml::IValue> value) {
  bool exists;
  size_t index;
  std::tie(exists, index) = Contains(input_names_, name);

  auto engine = m_session.as<LearningModelSession>()->GetEngine();
  winrt::com_ptr<_winml::IValue> device_value;
  WINML_THROW_IF_FAILED(engine->CreateOneInputAcrossDevices(name.c_str(), value.get(), device_value.put())
  );  // an input will always be copied on device mismatch

  if (exists) {
    inputs_[index] = device_value;
  } else {
    input_names_.push_back(name);
    inputs_.push_back(device_value);
  }

  return S_OK;
}

HRESULT LearningModelBinding::BindOutput(const std::string& name, winrt::com_ptr<_winml::IValue> value) {
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

std::vector<winrt::com_ptr<_winml::IValue>>& LearningModelBinding::GetOutputs() {
  return outputs_;
}

const std::vector<winrt::com_ptr<_winml::IValue>>& LearningModelBinding::GetInputs() const {
  return inputs_;
}

void LearningModelBinding::BindUnboundOutputs() {
  auto& bound_output_names = GetOutputNames();
  std::unordered_set<std::string> bound_output_names_set(bound_output_names.begin(), bound_output_names.end());

  // Get model output feature names
  auto model_impl = m_session.Model().as<winmlp::LearningModel>();
  auto output_features = model_impl->OutputFeatures();
  std::vector<ILearningModelFeatureDescriptor> output_descriptors(begin(output_features), end(output_features));

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
      return _winml::Strings::UTF8FromUnicode(p_name, size);
    }
  );

  // Find the set difference to determine if there are any unbound output features
  std::vector<std::string> unbound_output_names;
  std::copy_if(
    std::begin(output_feature_names),
    std::end(output_feature_names),
    std::inserter(unbound_output_names, std::begin(unbound_output_names)),
    [&](const auto& outputFeatureName) {
      return bound_output_names_set.find(outputFeatureName) == bound_output_names_set.end();
    }
  );

  // Add all unbound outputs to binding collection
  for (const auto& unbound_output : unbound_output_names) {
    auto engine = m_session.as<LearningModelSession>()->GetEngine();

    winrt::com_ptr<_winml::IValue> value;
    WINML_THROW_IF_FAILED(engine->CreateNullValue(value.put()));
    WINML_THROW_IF_FAILED(BindOutput(unbound_output, value));
  }
}

}  // namespace WINMLP
