// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "TensorKindFrom.h"

#include "MapFeatureDescriptor.h"
#include "TensorFeatureDescriptor.h"

namespace Windows::AI::MachineLearning {

//
// MapBase
//
// This is the base class for all data based Map types.
//
// Supported derived classes:
//    <String, Float>, <String, Int64>, <String, Double>, <String, String>
//    <Int64,  Float>, <Int64,  Int64>, <Int64,  Double>, <Int64,  String>
//
template <
    typename TDerived,
    typename TKey,
    typename TValue>
struct MapBase : winrt::implements<
                     MapBase<TDerived, TKey, TValue>,
                     winrt::Windows::AI::MachineLearning::ILearningModelFeatureValue,
                     WinML::IMapFeatureValue,
                     WinML::ILotusValueProviderPrivate> {
  static_assert(
      std::is_same<TKey, int64_t>::value ||
          std::is_same<TKey, winrt::hstring>::value,
      "Map keys must be int64_t or winrt::hstring!");

  static_assert(
      std::is_same<TValue, int64_t>::value ||
          std::is_same<TValue, double>::value ||
          std::is_same<TValue, float>::value ||
          std::is_same<TValue, winrt::hstring>::value,
      "Map values must be int64_t, double, float, or winrt::hstring!");

  using ABIMap = ::winrt::Windows::Foundation::Collections::IMap<TKey, TValue>;
  using ABIMapView = ::winrt::Windows::Foundation::Collections::IMapView<TKey, TValue>;

  MapBase(ABIMap const& data) : data_(data) {}

  static winrt::Windows::AI::MachineLearning::ILearningModelFeatureValue Create() {
    auto abiMap = winrt::single_threaded_map<TKey, TValue>();
    return winrt::make<TDerived>(abiMap);
  }

  static winrt::Windows::AI::MachineLearning::ILearningModelFeatureValue Create(const ABIMap& data) {
    return winrt::make<TDerived>(data);
  }

  static winrt::Windows::AI::MachineLearning::ILearningModelFeatureValue Create(const ABIMapView& data) {
    auto abiMap = winrt::single_threaded_map<TKey, TValue>();
    for (const auto& pair : data) {
      auto key = pair.Key();
      auto value = pair.Value();
      abiMap.Insert(key, value);
    }

    return winrt::make<TDerived>(abiMap);
  }
  // ILearningModelFeatureValue implementation
  winrt::Windows::AI::MachineLearning::LearningModelFeatureKind Kind() {
    return winrt::Windows::AI::MachineLearning::LearningModelFeatureKind::Map;
  }

  STDMETHOD(get_KeyKind)
  (winrt::Windows::AI::MachineLearning::TensorKind* kind) {
    FAIL_FAST_IF_NULL(kind);
    *kind = TensorKindFrom<TKey>::Type;
    return S_OK;
  }

  STDMETHOD(get_ValueDescriptor)
  (winrt::Windows::AI::MachineLearning::ILearningModelFeatureDescriptor* result) {
    FAIL_FAST_IF_NULL(result);

    *result = TensorFeatureDescriptorFrom<TValue>::CreateAnonymous(std::vector<int64_t>{});

    return S_OK;
  }

  STDMETHOD(GetValue)
  (WinML::BindingContext& context, IValue** out) {
    auto session = context.session.as<winrt::Windows::AI::MachineLearning::implementation::LearningModelSession>();
    auto engine = session->GetEngine();

    if (context.type == WinML::BindingType::kInput) {
      RETURN_IF_FAILED(engine->CreateMapValue(reinterpret_cast<::IInspectable*>(winrt::get_abi(data_)), TensorKindFrom<TKey>::Type, TensorKindFrom<TValue>::Type, out));
    } else {
      RETURN_IF_FAILED(engine->CreateNullValue(out));
    }
    return S_OK;
  }

  STDMETHOD(IsPlaceholder)
  (bool* pIsPlaceHolder) {
    FAIL_FAST_IF_NULL(pIsPlaceHolder);
    *pIsPlaceHolder = false;
    return S_OK;
  }

  STDMETHOD(UpdateSourceResourceData)
  (BindingContext& context, IValue* value) {
    data_.Clear();
    auto session = context.session.as<winrt::Windows::AI::MachineLearning::implementation::LearningModelSession>();
    auto engine = session->GetEngine();
    RETURN_IF_FAILED(engine->FillFromMapValue(reinterpret_cast<::IInspectable*>(winrt::get_abi(data_)), TensorKindFrom<TKey>::Type, TensorKindFrom<TValue>::Type, value));
    return S_OK;
  }

  STDMETHOD(AbiRepresentation)
  (
    winrt::Windows::Foundation::IInspectable& abiRepresentation) {
    data_.as(abiRepresentation);
    return S_OK;
  }

 private:
  ABIMap data_;
};

}  // namespace Windows::AI::MachineLearning
