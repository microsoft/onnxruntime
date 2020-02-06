// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
//    Map<String, Float>, Map<Int64, Float>
//
template <typename TDerived, typename T>
struct SequenceBase : public winrt::implements<
                          SequenceBase<TDerived, T>,
                          winml::ILearningModelFeatureValue,
                          WinML::ISequenceFeatureValue,
                          WinML::ILotusValueProviderPrivate> {
  using ABISequence = wfc::IIterable<T>;
  using AbiMapStringToFloat = wfc::IMap<winrt::hstring, float>;
  using AbiMapInt64BitToFloat = wfc::IMap<int64_t, float>;

  template <typename T> struct SequenceAbiTypeInfo {
    static constexpr winml::TensorKind Key = winml::TensorKind::Undefined;
    static constexpr winml::TensorKind Value = winml::TensorKind::Undefined;
  };
  template <> struct SequenceAbiTypeInfo<AbiMapStringToFloat> {
    static constexpr winml::TensorKind Key = winml::TensorKind::String;
    static constexpr winml::TensorKind Value = winml::TensorKind::Float;
  };
  template <>
  struct SequenceAbiTypeInfo<AbiMapInt64BitToFloat> {
    static constexpr winml::TensorKind Key = winml::TensorKind::Int64;
    static constexpr winml::TensorKind Value = winml::TensorKind::Float;
  };

  template <typename TElement>
  void
  GetElementDescriptor(winml::ILearningModelFeatureDescriptor* result) {
    static_assert(false, "Only sequences of of map<string, float> and map<int64, float> are supported.")
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

  STDMETHOD(GetValue)(
      WinML::BindingContext& context,
      IValue** out) {
    auto session = context.session.as<winrt::Windows::AI::MachineLearning::implementation::LearningModelSession>();
    auto engine = session->GetEngine();

    if (context.type == WinML::BindingType::kInput) {
	  // In opset 10, all ops that use sequences are seq<map>.
	  // In opset 11, we will need to support seq<tensor<t>> as well.
      RETURN_IF_FAILED(engine->CreateSequenceOfMapsValue(
		  reinterpret_cast<::IInspectable*>(winrt::get_abi(data_)),
          SequenceAbiTypeInfo<T>::Key, SequenceAbiTypeInfo<T>::Value, out));
    } else {
      RETURN_IF_FAILED(engine->CreateNullValue(out));
    }
    return S_OK;
  }

  STDMETHOD(IsPlaceholder)
  (
      bool* p_is_placeholder) {
    FAIL_FAST_IF_NULL(p_is_placeholder);
    *p_is_placeholder = false;
    return S_OK;
  }

  STDMETHOD(UpdateSourceResourceData)(
      BindingContext& context,
      IValue* out) {
    auto writable_vector = data_.as<wfc::IVector<T>>();
    writable_vector.Clear();

    auto session = context.session.as<winrt::Windows::AI::MachineLearning::implementation::LearningModelSession>();
    auto engine = session->GetEngine();
    RETURN_IF_FAILED(engine->FillSequenceOfMapsValue(reinterpret_cast<::IInspectable*>(winrt::get_abi(data_)), SequenceAbiTypeInfo<T>::Key, SequenceAbiTypeInfo<T>::Value, out));

    return S_OK;
  }

  STDMETHOD(AbiRepresentation)(
    wf::IInspectable& abi_representation) {
    data_.as(abi_representation);
    return S_OK;
  }

 private:
  ABISequence data_;
};

}  // namespace Windows::AI::MachineLearning