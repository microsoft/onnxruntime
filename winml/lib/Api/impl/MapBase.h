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
  using LotusMap = std::pair<std::vector<LotusKey>, std::vector<LotusValue>>;
  using ABIMap = ::winrt::Windows::Foundation::Collections::IMap<TKey, TValue>;
  using ABIMapView = ::winrt::Windows::Foundation::Collections::IMapView<TKey, TValue>;

  //template <typename TRawType>
  //static std::vector<TRawType> ConvertToABIType(Ort::Value& ort_value) {
  //  // make sure this is an array of these types
  //  auto shape = ort_value.GetTensorTypeAndShapeInfo().GetShape();
  //  // there needs to be only one dimension
  //  THROW_HR_IF(E_INVALIDARG, shape.size() != 1);
  //  auto lotus_value = ort_value.GetTensorMutableData<typename ValidLotusType<TRawType>::Type>();
  //  // now go through all the entries
  //  std::vector<TRawType> out;
  //  for (auto i = 0; i < shape[0]; i++) {
  //    out.push_back(lotus_value[i]);
  //  }
  //  // retun the vector
  //  return out;
  //}

  //template <>
  //static std::vector<winrt::hstring> ConvertToABIType<winrt::hstring>(Ort::Value& ort_value) {
  //  auto strings = ort_value.GetStrings();
  //  std::vector<winrt::hstring> out;
  //  for (auto i = 0; i < strings.size(); ++i) {
  //    out.push_back(WinML::Strings::HStringFromUTF8(strings[i].c_str()));
  //  }
  //  return out;
  //}

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

  //template <typename TLotusKey, typename TLotusValue>
  //static onnxruntime::MLDataType GetLotusType(winmla::IWinMLAdapter* adapter) {
  //  return adapter->GetMapType(TensorKindFrom<TLotusKey>::Type, TensorKindFrom<TLotusValue>::Type);
  //}

  STDMETHOD(GetValue)
  (WinML::BindingContext& context, IValue** out) {
    ORT_UNUSED_PARAMETER(context);
    auto session = context.session.as<winrt::Windows::AI::MachineLearning::implementation::LearningModelSession>();
    auto engine = session->GetEngine();

    if (context.type == WinML::BindingType::kInput) {
      auto map = FillLotusMapFromAbiMap(data_);
      RETURN_IF_FAILED(engine->CreateMapValue(winrt::get_abi(data_), TensorKindFrom<TKey>::Type, TensorKindFrom<TValue>::Type, out));
      RETURN_IF_FAILED(engine->CreateMapValue(map->first.data(), map->second.data(), map->first.size(), out));

      // now create OrtValue wrappers over the buffers
      auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
      std::vector<int64_t> shape = {static_cast<int64_t>(len)};
      auto keys_ort_value = Ort::Value::CreateTensor<TLotusKey>(cpu_memory, keys, len, shape.data(), shape.size());
      auto values_ort_value = Ort::Value::CreateTensor<TLotusValue>(cpu_memory, values, len, shape.data(), shape.size());
      // make the map
      return Ort::Value::CreateMap(keys_ort_value, values_ort_value);
    } else {
      engine->CreateNullValue(out);
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
    ORT_UNUSED_PARAMETER(context);
    ORT_UNUSED_PARAMETER(value);
    data_.Clear();

    //Ort::AllocatorWithDefaultOptions allocator;

    //// get the keys
    //OrtValue* ptr = nullptr;
    //Ort::ThrowOnError(Ort::GetApi().GetValue(ort_value, 0, allocator, &ptr));
    //Ort::Value keys{ptr};
    //// get the values
    //ptr = nullptr;
    //Ort::ThrowOnError(Ort::GetApi().GetValue(ort_value, 1, allocator, &ptr));
    //Ort::Value values{ptr};

    //auto keys_vector = ConvertToABIType<TKey>(keys);
    //auto values_vector = ConvertToABIType<TValue>(values);

    //auto len = keys.GetCount();
    //for (auto i = 0; i < len; ++i) {
    //  data_.Insert(keys_vector[i], values_vector[i]);
    //}
    return S_OK;

    // TODO: code this
    //const LotusMap& map = *static_cast<LotusMap*>(pResource);
    //for (const auto& pair : map) {
    //  auto key = ConvertToABIType<TKey>(pair.first);
    //  auto value = ConvertToABIType<TValue>(pair.second);
    //  data_.Insert(key, value);
    //}

    //return S_OK;
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
