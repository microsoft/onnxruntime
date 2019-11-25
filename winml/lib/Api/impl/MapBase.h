// Copyright (c) Microsoft Corporation.
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

  template <typename T>
  struct ValidLotusType { using Type = T; };
  template <>
  struct ValidLotusType<winrt::hstring> { using Type = std::string; };

  using LotusKey = typename ValidLotusType<TKey>::Type;
  using LotusValue = typename ValidLotusType<TValue>::Type;
  //using LotusMap = std::map<LotusKey, LotusValue>;
  using LotusMap = std::pair<std::vector<LotusKey>, std::vector<LotusValue>>;
  using ABIMap = ::winrt::Windows::Foundation::Collections::IMap<TKey, TValue>;
  using ABIMapView = ::winrt::Windows::Foundation::Collections::IMapView<TKey, TValue>;

  template <typename TRawType>
  static typename ValidLotusType<TRawType>::Type ConvertToValidLotusType(TRawType raw) {
    return raw;
  }

  template <>
  static typename ValidLotusType<winrt::hstring>::Type ConvertToValidLotusType(winrt::hstring raw) {
    return WinML::Strings::UTF8FromHString(raw);
  }

  template <typename TRawType>
  static TRawType ConvertToABIType(const typename ValidLotusType<TRawType>::Type& lotusValue) {
    TRawType copy = lotusValue;
    return copy;
  }

  template <>
  static typename winrt::hstring ConvertToABIType(const typename ValidLotusType<winrt::hstring>::Type& lotusValue) {
    return WinML::Strings::HStringFromUTF8(lotusValue);
  }

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

  void ConvertToLotusMap(const ABIMap& map) {
    std::vector<LotusKey> keys;
    std::vector<LotusValue> values;
    for (const auto& pair : map) {
      auto key = ConvertToValidLotusType(pair.Key());
      auto value = ConvertToValidLotusType(pair.Value());
      keys.push_back(key);
      values.push_back(value);
    }
    lotus_data_ = std::make_unique<LotusMap>(std::make_pair(keys, values));
  }

  template <typename TLotusKey, typename TLotusValue>
  static onnxruntime::MLDataType GetLotusType(_winmla::IWinMLAdapter* adapter) {
    return adapter->GetMapType(TensorKindFrom<TLotusKey>::Type, TensorKindFrom<TLotusValue>::Type);
  }

  template <typename TLotusKey, typename TLotusValue>
  static Ort::Value CreateOrtMap(TLotusKey * keys, TLotusValue * values, size_t len) {
    // now create OrtValue wrappers over the buffers
    auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {static_cast<int64_t>(len)};
    auto keys_ort_value = Ort::Value::CreateTensor<TLotusKey>(cpu_memory, keys, len, shape.data(), shape.size());
    auto values_ort_value = Ort::Value::CreateTensor<TLotusValue>(cpu_memory, values, len, shape.data(), shape.size());
    // make the map
    return Ort::Value::CreateMap(keys_ort_value, values_ort_value);
  }

  STDMETHOD(GetOrtValue)
  (WinML::BindingContext& context, OrtValue** ort_value) {
    // TODO: Tensorized data should be cached so multiple bindings work more efficiently

    // TODO : we need to handle inputs.   for now only handle outputs and don't pre allocate anything
    if (context.type == WinML::BindingType::kOutput) {
      return S_OK;
    }

    // handle inputs, create and store a copy of the map
    ConvertToLotusMap(data_);

    // and make the map
    *ort_value = CreateOrtMap(lotus_data_->first.data(), lotus_data_->second.data(), lotus_data_->first.size());
    return S_OK;
  }

  STDMETHOD(IsPlaceholder)
  (bool* pIsPlaceHolder) {
    FAIL_FAST_IF_NULL(pIsPlaceHolder);
    *pIsPlaceHolder = false;
    return S_OK;
  }

  STDMETHOD(UpdateSourceResourceData)
  (BindingContext& context, OrtValue* ort_value) {
    data_.Clear();

    void* pResource = nullptr;
    Ort::ThrowOnError(Ort::GetApi().GetTensorMutableData(ort_value, &pResource));

    // TODO: code this
    //const LotusMap& map = *static_cast<LotusMap*>(pResource);
    //for (const auto& pair : map) {
    //  auto key = ConvertToABIType<TKey>(pair.first);
    //  auto value = ConvertToABIType<TValue>(pair.second);
    //  data_.Insert(key, value);
    //}

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
  std::unique_ptr<LotusMap> lotus_data_;
};

}  // namespace Windows::AI::MachineLearning
