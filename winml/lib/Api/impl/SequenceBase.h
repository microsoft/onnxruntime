#pragma once

#include "MapFeatureDescriptor.h"
#include "SequenceFeatureDescriptor.h"
#include "TensorFeatureDescriptor.h"

namespace Windows::AI::MachineLearning {

// SequenceBase
//
// This is the base class for all data based Sequence types.
//
// Supported derived classes:
//    Float, Int64, Double, String, Map<String, Float>, Map<Int64, Float>
//
template <typename TDerived, typename T>
struct SequenceBase : public winrt::implements<
                          SequenceBase<TDerived, T>,
                          winml::ILearningModelFeatureValue,
                          WinML::ISequenceFeatureValue,
                          WinML::ILotusValueProviderPrivate> {
  using AbiMapStringToFloat = wfc::IMap<winrt::hstring, float>;
  using AbiMapInt64BitToFloat = wfc::IMap<int64_t, float>;

  template <typename T>
  struct ValidLotusType { using Type = T; };
  template <>
  struct ValidLotusType<winrt::hstring> { using Type = std::string; };
  template <>
  struct ValidLotusType<AbiMapStringToFloat> { using Type = std::map<std::string, float>; };
  template <>
  struct ValidLotusType<AbiMapInt64BitToFloat> { using Type = std::map<int64_t, float>; };

  template <typename TElement>
  void
  GetElementDescriptor(winml::ILearningModelFeatureDescriptor* result) {
    *result = TensorFeatureDescriptorFrom<T>::CreateAnonymous(
        std::vector<int64_t>{1, 1, 1, 1});
  }

  template <>
  void
  GetElementDescriptor<wfc::IMap<winrt::hstring, float>>(
      winml::ILearningModelFeatureDescriptor* result) {
    // zero dimensional tensor has empty shape
    auto value_descriptor =
        WinML::TensorFeatureDescriptorFrom<float>::CreateAnonymous(
            std::vector<int64_t>{});
    *result =
        winrt::make<winmlp::MapFeatureDescriptor>(
            nullptr /* set to null as values are name-less */,
            nullptr /* set to null as values are description-less */,
            false /* set to false as values dont have required annotations */,
            winml::TensorKind::String /* key kind */,
            value_descriptor /* value kind */);
  }

  template <>
  void
  GetElementDescriptor<wfc::IMap<int64_t, float>>(
      winml::ILearningModelFeatureDescriptor* result) {
    // zero dimensional tensor has empty shape
    auto value_descriptor =
        WinML::TensorFeatureDescriptorFrom<float>::CreateAnonymous(
            std::vector<int64_t>{});
    *result =
        winrt::make<winmlp::MapFeatureDescriptor>(
            nullptr /* set to null as values are name-less */,
            nullptr /* set to null as values are description-less */,
            false /* set to false as values dont have required annotations */,
            winml::TensorKind::Int64 /* key kind */,
            value_descriptor /* value kind */);
  }

  using LotusSequence = std::vector<typename ValidLotusType<T>::Type>;
  using ABISequence = wfc::IIterable<T>;

  SequenceBase(const ABISequence& data) : data_(data) {}

  static winml::ILearningModelFeatureValue
  Create() {
    auto sequence = winrt::single_threaded_vector<T>();
    return winrt::make<TDerived>(sequence);
  }

  static winml::ILearningModelFeatureValue
  Create(
      const ABISequence& data) {
    return winrt::make<TDerived>(data);
  }

  // ILearningModelFeatureValue implementation
  winml::LearningModelFeatureKind
  Kind() {
    return winml::LearningModelFeatureKind::Sequence;
  }

  STDMETHOD(get_ElementDescriptor)
  (
      winml::ILearningModelFeatureDescriptor* result) {
    FAIL_FAST_IF_NULL(result);

    GetElementDescriptor<T>(result);

    return S_OK;
  }

  template <typename TRawType>
  static
      typename ValidLotusType<TRawType>::Type
      ConvertToValidLotusType(
          TRawType raw) {
    return raw;
  }

  template <>
  static
      typename ValidLotusType<winrt::hstring>::Type
      ConvertToValidLotusType(
          winrt::hstring raw) {
    return WinML::Strings::UTF8FromHString(raw);
  }

  template <>
  static
      typename ValidLotusType<AbiMapStringToFloat>::Type
      ConvertToValidLotusType(
          AbiMapStringToFloat raw) {
    std::map<std::string, float> lotus_map;
    for (auto pair : raw) {
      auto key = WinML::Strings::UTF8FromHString(pair.Key());
      lotus_map[key] = pair.Value();
    }
    return lotus_map;
  }

  template <>
  static
      typename ValidLotusType<AbiMapInt64BitToFloat>::Type
      ConvertToValidLotusType(
          AbiMapInt64BitToFloat raw) {
    std::map<int64_t, float> lotus_map;
    for (const auto& pair : raw) {
      lotus_map[pair.Key()] = pair.Value();
    }
    return lotus_map;
  }

  static LotusSequence
  ConvertToLotusSequence(
      const ABISequence& sequence) {
    LotusSequence lotus_sequence;

    std::transform(
        begin(sequence),
        end(sequence),
        std::back_inserter(lotus_sequence),
        [](const auto& value) {
          return ConvertToValidLotusType(value);
        });

    return lotus_sequence;
  }

  STDMETHOD(GetOrtValue)
  (
      WinML::BindingContext& context,
      OrtValue* ml_value) {
    // TODO: Tensorized data should be cached so multiple bindings work more efficiently

    // Create a copy of the sequence
    auto sequence = context.type == WinML::BindingType::kInput
                        ? std::make_unique<LotusSequence>(ConvertToLotusSequence(data_))
                        : std::make_unique<LotusSequence>();

    OrtValue value;
    value.Init(
        sequence.release(),
        onnxruntime::DataTypeImpl::GetType<LotusSequence>(),
        onnxruntime::DataTypeImpl::GetType<LotusSequence>()->GetDeleteFunc());

    *ml_value = value;

    return S_OK;
  }

  STDMETHOD(IsPlaceholder)
  (
      bool* p_is_placeholder) {
    FAIL_FAST_IF_NULL(p_is_placeholder);
    *p_is_placeholder = false;
    return S_OK;
  }

  template <typename TRawType>
  static TRawType
  ConvertToABIType(
      typename ValidLotusType<TRawType>::Type lotus_value) {
    return lotus_value;
  }

  template <>
  static winrt::hstring
  ConvertToABIType(
      typename ValidLotusType<winrt::hstring>::Type lotus_value) {
    return WinML::Strings::HStringFromUTF8(lotus_value);
  }

  template <>
  static AbiMapStringToFloat
  ConvertToABIType(
      typename ValidLotusType<AbiMapStringToFloat>::Type lotus_value) {
    std::map<winrt::hstring, float> copy;
    for (const auto& pair : lotus_value) {
      auto key = WinML::Strings::HStringFromUTF8(pair.first);
      copy[key] = pair.second;
    }
    return winrt::single_threaded_map<winrt::hstring, float>(
        std::move(copy));
  }

  template <>
  static AbiMapInt64BitToFloat
  ConvertToABIType(
      typename ValidLotusType<AbiMapInt64BitToFloat>::Type lotus_value) {
    return winrt::single_threaded_map<int64_t, float>(
        std::move(lotus_value));
  }

  STDMETHOD(UpdateSourceResourceData)
  (
      BindingContext& context,
      OrtValue& ml_value) {
    auto writable_vector = data_.as<wfc::IVector<T>>();

    writable_vector.Clear();

    const auto& sequence = ml_value.Get<LotusSequence>();

    for (const auto& element : sequence) {
      writable_vector.Append(ConvertToABIType<T>(element));
    }

    return S_OK;
  }

  STDMETHOD(AbiRepresentation)
  (
      wf::IInspectable& abi_representation) {
    data_.as(abi_representation);
    return S_OK;
  }

 private:
  ABISequence data_;
};

}  // namespace Windows::AI::MachineLearning
