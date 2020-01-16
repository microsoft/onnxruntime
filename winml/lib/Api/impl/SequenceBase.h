// Copyright (c) Microsoft Corporation.
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
  using AbiMapStringToFloat = wfc::IMap<winrt::hstring, float>;
  using AbiMapInt64BitToFloat = wfc::IMap<int64_t, float>;

  template <typename T>
  struct ValidLotusType { using Type = T; };
  template <>
  struct ValidLotusType<AbiMapStringToFloat> {
    //using Type = std::map<std::string, float>;
    using TKey = std::string;
    using TValue = float;
    using Type = std::pair<std::vector<TKey>, std::vector<TValue>>;
    using ABIKey = winrt::hstring;
    using ABIValue = TValue;
  };
  template <>
  struct ValidLotusType<AbiMapInt64BitToFloat> {
    //using Type = std::map<int64_t, float>;
    using TKey = int64_t;
    using TValue = float;
    using Type = std::pair<std::vector<TKey>, std::vector<TValue>>;
    using ABIKey = TKey;
    using ABIValue = TValue;
  };

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
    std::vector<ValidLotusType<AbiMapStringToFloat>::TKey> keys;
    std::vector<ValidLotusType<AbiMapStringToFloat>::TValue> values;
    for (auto pair : raw) {
      auto key = WinML::Strings::UTF8FromHString(pair.Key());
      keys.push_back(key);
      values.push_back(pair.Value());
    }
    return std::make_pair(keys, values);
  }

  template <>
  static
      typename ValidLotusType<AbiMapInt64BitToFloat>::Type
      ConvertToValidLotusType(
          AbiMapInt64BitToFloat raw) {
    std::vector<ValidLotusType<AbiMapInt64BitToFloat>::TKey> keys;
    std::vector<ValidLotusType<AbiMapInt64BitToFloat>::TValue> values;
    for (const auto& pair : raw) {
      keys.push_back(pair.Key());
      values.push_back(pair.Value());
    }
    return std::make_pair(keys, values);
  }

  void
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

    lotus_data_ = std::make_unique<LotusSequence>(lotus_sequence);
  }

  template <typename TLotusKey, typename TLotusValue>
  static Ort::Value CreateOrtMap(TLotusKey* keys, TLotusValue* values, size_t len) {
    // now create OrtValue wrappers over the buffers
    auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {static_cast<int64_t>(len)};
    auto keys_ort_value = Ort::Value::CreateTensor<TLotusKey>(cpu_memory, keys, len, shape.data(), shape.size());
    auto values_ort_value = Ort::Value::CreateTensor<TLotusValue>(cpu_memory, values, len, shape.data(), shape.size());
    // make the map
    return Ort::Value::CreateMap(keys_ort_value, values_ort_value);
  }

  STDMETHOD(GetValue)(
      WinML::BindingContext& context,
      IValue** out) {
    // TODO: Tensorized data should be cached so multiple bindings work more efficiently

    // TODO : we need to handle inputs.   for now only handle outputs and don't pre allocate anything
    if (context.type == WinML::BindingType::kOutput) {
      *out = nullptr;
      return S_OK;
    }

    // handle inputs, create and store a copy of the sequence
    ConvertToLotusSequence(data_);

    // now create OrtValue wrappers over the buffers
    std::vector<Ort::Value> sequence_values;
    for (auto it = lotus_data_->begin(); it != lotus_data_->end(); ++it) {
      // make a ort value for this map
      auto map = *it;
      //sequence_values.emplace_back(CreateOrtMap(map.first.data(), map.second.data(), map.first.size()));
    }
    //*ort_value = Ort::Value::CreateSequence(sequence_values).release();
    return S_OK;

    /*    winrt::com_ptr<winmla::IWinMLAdapter> adapter;
    RETURN_IF_FAILED(OrtGetWinMLAdapter(adapter.put()));
    auto lotus_type = adapter->GetVectorMapType(
        TensorKindFrom<ValidLotusType<T>::TKey>::Type, 
        TensorKindFrom<ValidLotusType<T>::TValue>::Type);

    winrt::com_ptr<winmla::IOrtValue> ml_value_out;
    adapter->CreateOrtValue(lotus_data_.get(), lotus_type, ml_value_out.put());

    *ml_value = ml_value_out.detach();*/
  }

  STDMETHOD(IsPlaceholder)
  (
      bool* p_is_placeholder) {
    FAIL_FAST_IF_NULL(p_is_placeholder);
    *p_is_placeholder = false;
    return S_OK;
  }

  template <typename TRawType>
  static std::vector<TRawType> ConvertToABIType(Ort::Value& ort_value) {
    // make sure this is an array of these types
    auto shape = ort_value.GetTensorTypeAndShapeInfo().GetShape();
    // there needs to be only one dimension
    THROW_HR_IF(E_INVALIDARG, shape.size() != 1);
    auto lotus_value = ort_value.GetTensorMutableData<typename ValidLotusType<TRawType>::Type>();
    // now go through all the entries
    std::vector<TRawType> out;
    for (auto i = 0; i < shape[0]; i++) {
      out.push_back(lotus_value[i]);
    }
    // return the vector
    return out;
  }

  template <>
  static std::vector<winrt::hstring> ConvertToABIType<winrt::hstring>(Ort::Value& ort_value) {
    auto strings = ort_value.GetStrings();
    std::vector<winrt::hstring> out;
    for (auto i = 0; i < strings.size(); ++i) {
      out.push_back(WinML::Strings::HStringFromUTF8(strings[i].c_str()));
    }
    return out;
  }

  STDMETHOD(UpdateSourceResourceData)(
      BindingContext& context,
      IValue* out) {
    ORT_UNUSED_PARAMETER(out);
    ORT_UNUSED_PARAMETER(context);
    /*auto writable_vector = data_.as<wfc::IVector<T>>();
    writable_vector.Clear();

    Ort::AllocatorWithDefaultOptions allocator;
    size_t len;
    Ort::ThrowOnError(Ort::GetApi().GetValueCount(ort_value, &len));
    for (auto i = 0; i < len; ++i) {
      OrtValue* out = nullptr;
      Ort::ThrowOnError(Ort::GetApi().GetValue(ort_value, i, allocator, &out));
      Ort::Value map{out};
      auto keys = map.GetValue(0, allocator);
      auto values = map.GetValue(1, allocator);

      auto keys_vector = ConvertToABIType<typename ValidLotusType<T>::ABIKey>(keys);
      auto values_vector = ConvertToABIType<typename ValidLotusType<T>::ABIValue>(values);

      std::map<typename ValidLotusType<T>::ABIKey, typename ValidLotusType<T>::ABIValue> std_map;
      for (auto j = 0; j < keys_vector.size(); ++j) {
        std_map[keys_vector[j]] = values_vector[j];
      }
      auto abi_map = winrt::single_threaded_map<typename ValidLotusType<T>::ABIKey, typename ValidLotusType<T>::ABIValue>(
        std::move(std_map));
    
      writable_vector.Append(abi_map);
    }*/
    return S_OK;
  }

  STDMETHOD(AbiRepresentation)(
    wf::IInspectable& abi_representation) {
    data_.as(abi_representation);
    return S_OK;
  }

 private:
  ABISequence data_;
  std::unique_ptr<LotusSequence> lotus_data_;
};

}  // namespace Windows::AI::MachineLearning
