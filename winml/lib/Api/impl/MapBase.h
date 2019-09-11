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
  using LotusMap = std::map<LotusKey, LotusValue>;
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
  static TRawType ConvertToABIType(typename ValidLotusType<TRawType>::Type lotusValue) {
    return lotusValue;
  }

  template <>
  static typename winrt::hstring ConvertToABIType(typename ValidLotusType<winrt::hstring>::Type lotusValue) {
    return WinML::Strings::HStringFromUTF8(lotusValue);
  }

  MapBase(ABIMap const& data) : m_data(data) {}

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

  static LotusMap ConvertToLotusMap(const ABIMap& map) {
    LotusMap lotusMap;
    for (const auto& pair : map) {
      auto key = ConvertToValidLotusType(pair.Key());
      auto value = ConvertToValidLotusType(pair.Value());
      lotusMap[key] = value;
    }
    return lotusMap;
  }

  STDMETHOD(GetOrtValue)
  (WinML::BindingContext& context, OrtValue* mlValue) {
    // TODO: Tensorized data should be cached so multiple bindings work more efficiently

    // Create a copy of the map
    auto map = context.type == WinML::BindingType::kInput ? std::make_unique<LotusMap>(ConvertToLotusMap(m_data)) : std::make_unique<LotusMap>();

    OrtValue value;
    value.Init(
        map.release(),
        onnxruntime::DataTypeImpl::GetType<LotusMap>(),
        onnxruntime::DataTypeImpl::GetType<LotusMap>()->GetDeleteFunc());

    *mlValue = value;
    return S_OK;
  }

  STDMETHOD(IsPlaceholder)
  (bool* pIsPlaceHolder) {
    FAIL_FAST_IF_NULL(pIsPlaceHolder);
    *pIsPlaceHolder = false;
    return S_OK;
  }

  STDMETHOD(UpdateSourceResourceData)
  (BindingContext& context, OrtValue& mlValue) {
    m_data.Clear();

    const auto& map = mlValue.Get<LotusMap>();

    for (const auto& pair : map) {
      auto key = ConvertToABIType<TKey>(pair.first);
      auto value = ConvertToABIType<TValue>(pair.second);
      m_data.Insert(key, value);
    }

    return S_OK;
  }

  STDMETHOD(AbiRepresentation)
  (
      winrt::Windows::Foundation::IInspectable& abiRepresentation) {
    m_data.as(abiRepresentation);
    return S_OK;
  }

 private:
  ABIMap m_data;
};

}  // namespace Windows::AI::MachineLearning
