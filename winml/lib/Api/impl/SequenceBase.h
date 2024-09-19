// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "MapFeatureDescriptor.h"
#include "SequenceFeatureDescriptor.h"
#include "TensorFeatureDescriptor.h"
#include "LearningModelSession.h"
#include "ISequenceFeatureValue.h"

#include "FeatureValues.h"

namespace _winml {

// SequenceBase
//
// This is the base class for all data based Sequence types.
//
// Supported derived classes:
//    Map<String, Float>, Map<Int64, Float>
//
template <typename TDerived, typename T, typename TRaw>
struct SequenceBase : public winrt::implements<
                        SequenceBase<TDerived, T, TRaw>,
                        winml::ILearningModelFeatureValue,
                        _winml::ISequenceFeatureValue,
                        _winml::ILotusValueProviderPrivate> {
  using ABISequence = wfc::IIterable<T>;
  using AbiMapStringToFloat = wfc::IMap<winrt::hstring, float>;
  using AbiMapInt64BitToFloat = wfc::IMap<int64_t, float>;

  static_assert(
    std::is_same<T, AbiMapStringToFloat>::value || std::is_same<T, AbiMapInt64BitToFloat>::value ||
      std::is_same<TRaw, bool>::value || std::is_same<TRaw, float>::value || std::is_same<TRaw, double>::value ||
      std::is_same<TRaw, int8_t>::value || std::is_same<TRaw, uint8_t>::value || std::is_same<TRaw, uint16_t>::value ||
      std::is_same<TRaw, int16_t>::value || std::is_same<TRaw, uint32_t>::value || std::is_same<TRaw, int32_t>::value ||
      std::is_same<TRaw, uint64_t>::value || std::is_same<TRaw, int64_t>::value ||
      std::is_same<TRaw, _winml::Half>::value || std::is_same<TRaw, std::string>::value,
    "Only sequences of of map<string, float>, map<int64, float> and tensor<T> are supported."
  );

  template <typename T>
  struct SequenceAbiTypeInfo {
    static constexpr winml::TensorKind Key = winml::TensorKind::Undefined;
    static constexpr winml::TensorKind Value = winml::TensorKind::Undefined;
  };
  template <>
  struct SequenceAbiTypeInfo<AbiMapStringToFloat> {
    static constexpr winml::TensorKind Key = winml::TensorKind::String;
    static constexpr winml::TensorKind Value = winml::TensorKind::Float;
  };
  template <>
  struct SequenceAbiTypeInfo<AbiMapInt64BitToFloat> {
    static constexpr winml::TensorKind Key = winml::TensorKind::Int64;
    static constexpr winml::TensorKind Value = winml::TensorKind::Float;
  };

  template <typename TElement>
  void GetElementDescriptor(winml::ILearningModelFeatureDescriptor* result) {
    *result = _winml::TensorFeatureDescriptorFrom<TRaw>::CreateAnonymous(std::vector<int64_t>{});
  }

  template <>
  void GetElementDescriptor<wfc::IMap<winrt::hstring, float>>(winml::ILearningModelFeatureDescriptor* result) {
    // zero dimensional tensor has empty shape
    auto value_descriptor = _winml::TensorFeatureDescriptorFrom<float>::CreateAnonymous(std::vector<int64_t>{});
    *result = winrt::make<winmlp::MapFeatureDescriptor>(
      nullptr /* set to null as values are name-less */,
      nullptr /* set to null as values are description-less */,
      false /* set to false as values dont have required annotations */,
      winml::TensorKind::String /* key kind */,
      value_descriptor /* value kind */
    );
  }

  template <>
  void GetElementDescriptor<wfc::IMap<int64_t, float>>(winml::ILearningModelFeatureDescriptor* result) {
    // zero dimensional tensor has empty shape
    auto value_descriptor = _winml::TensorFeatureDescriptorFrom<float>::CreateAnonymous(std::vector<int64_t>{});
    *result = winrt::make<winmlp::MapFeatureDescriptor>(
      nullptr /* set to null as values are name-less */,
      nullptr /* set to null as values are description-less */,
      false /* set to false as values dont have required annotations */,
      winml::TensorKind::Int64 /* key kind */,
      value_descriptor /* value kind */
    );
  }

  SequenceBase(const ABISequence& data) : data_(data) {}

  static winml::ILearningModelFeatureValue Create() {
    auto sequence = winrt::single_threaded_vector<T>();
    return winrt::make<TDerived>(sequence);
  }

  static winml::ILearningModelFeatureValue Create(const ABISequence& data) { return winrt::make<TDerived>(data); }

  // ILearningModelFeatureValue implementation
  winml::LearningModelFeatureKind Kind() { return winml::LearningModelFeatureKind::Sequence; }

  STDMETHOD(get_ElementDescriptor)(winml::ILearningModelFeatureDescriptor* result) {
    FAIL_FAST_IF_NULL(result);

    GetElementDescriptor<T>(result);

    return S_OK;
  }

  STDMETHOD(GetValue)(_winml::BindingContext& context, IValue** out) {
    auto session = context.session.as<winmlp::LearningModelSession>();
    auto engine = session->GetEngine();

    if (context.type == _winml::BindingType::kInput) {
      winml::ILearningModelFeatureDescriptor descriptor(nullptr);
      GetElementDescriptor<T>(&descriptor);

      if (descriptor.Kind() == winml::LearningModelFeatureKind::Map) {
        // In opset 10 and earlier only seq<map<,>> were supported
        RETURN_IF_FAILED(engine->CreateSequenceOfMapsValue(
          reinterpret_cast<::IInspectable*>(winrt::get_abi(data_)),
          SequenceAbiTypeInfo<T>::Key,
          SequenceAbiTypeInfo<T>::Value,
          out
        ));
      } else if (descriptor.Kind() == winml::LearningModelFeatureKind::Tensor) {
        // In opset 11, operators that require seq<tensor<t>> were added.

        // IVector<Tensor*> -> std::vector<IValue>
        //
        // Convert all of the data in the sequence of tensors IVector into the appropriate
        // IValues based on the session's EP. This is done by calling into each tensor's
        // GetValue and delegating tensorization to each of those objects.
        //
        // The resulting tensors are collected into a vector.
        std::vector<winrt::com_ptr<_winml::IValue>> sequence;
        for (auto tensor : data_) {
          auto value_provider = tensor.as<_winml::ILotusValueProviderPrivate>();
          winrt::com_ptr<_winml::IValue> out_value;
          RETURN_IF_FAILED(value_provider->GetValue(context, out_value.put()));
          sequence.push_back(out_value);
        }

        // The collection of IValues needs wrapped into a single IValue
        // which represents the sequence<tensor> value.
        std::vector<_winml::IValue*> sequence_values;
        std::transform(std::begin(sequence), std::end(sequence), std::back_inserter(sequence_values), [](auto value) {
          return value.get();
        });

        RETURN_IF_FAILED(engine->CreateSequenceOfValuesValue(sequence_values.data(), sequence_values.size(), out));
      } else {
        // This should never happen, as the static_assert at the beginning of the code should prevent this path
        // from even being hit.
        FAIL_FAST();
      }
    } else {
      RETURN_IF_FAILED(engine->CreateNullValue(out));
    }
    return S_OK;
  }

  STDMETHOD(IsPlaceholder)
  (bool* p_is_placeholder) {
    FAIL_FAST_IF_NULL(p_is_placeholder);
    *p_is_placeholder = false;
    return S_OK;
  }

  template <typename TElement = T>
  auto CreatePlaceholderTensor() {
    return TElement(nullptr);
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorBoolean>() {
    return winml::TensorBoolean::Create();
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorFloat>() {
    return winml::TensorFloat::Create();
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorDouble>() {
    return winml::TensorDouble::Create();
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorInt8Bit>() {
    return winml::TensorInt8Bit::Create();
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorUInt8Bit>() {
    return winml::TensorUInt8Bit::Create();
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorUInt16Bit>() {
    return winml::TensorUInt16Bit::Create();
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorInt16Bit>() {
    return winml::TensorInt16Bit::Create();
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorUInt32Bit>() {
    return winml::TensorUInt32Bit::Create();
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorInt32Bit>() {
    return winml::TensorInt32Bit::Create();
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorUInt64Bit>() {
    return winml::TensorUInt64Bit::Create();
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorInt64Bit>() {
    return winml::TensorInt64Bit::Create();
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorFloat16Bit>() {
    return winml::TensorFloat16Bit::Create();
  }
  template <>
  auto CreatePlaceholderTensor<winml::TensorString>() {
    return winml::TensorString::Create();
  }

  void AppendValue(_winml::BindingContext& context, wfc::IVector<T> data, winrt::com_ptr<_winml::IValue> value) {
    auto tensor = CreatePlaceholderTensor();
    auto value_provider = tensor.as<_winml::ILotusValueProviderPrivate>();
    WINML_THROW_IF_FAILED(value_provider->UpdateSourceResourceData(context, value.get()));
    data.Append(tensor);
  }

  STDMETHOD(UpdateSourceResourceData)
  (BindingContext& context, IValue* out) {
    auto writable_vector = data_.as<wfc::IVector<T>>();
    writable_vector.Clear();

    auto session = context.session.as<winmlp::LearningModelSession>();
    auto engine = session->GetEngine();

    winml::ILearningModelFeatureDescriptor descriptor(nullptr);
    GetElementDescriptor<T>(&descriptor);

    if (descriptor.Kind() == winml::LearningModelFeatureKind::Map) {
      // In opset 10 and earlier only seq<map<,>> were supported
      RETURN_IF_FAILED(engine->FillSequenceOfMapsValue(
        reinterpret_cast<::IInspectable*>(winrt::get_abi(data_)),
        SequenceAbiTypeInfo<T>::Key,
        SequenceAbiTypeInfo<T>::Value,
        out
      ));
    } else if (descriptor.Kind() == winml::LearningModelFeatureKind::Tensor) {
      // In opset 11, operators that require seq<tensor<t>> were added.
      std::vector<winrt::com_ptr<_winml::IValue>> tensor_values;
      RETURN_IF_FAILED(engine->GetSequenceOfTensorValues(out, tensor_values));

      for (auto tensor_value : tensor_values) {
        AppendValue(context, writable_vector, tensor_value);
      }
    } else {
      FAIL_FAST();
    }
    return S_OK;
  }

  STDMETHOD(AbiRepresentation)
  (wf::IInspectable& abi_representation) {
    data_.as(abi_representation);
    return S_OK;
  }

 private:
  ABISequence data_;
};

}  // namespace _winml
