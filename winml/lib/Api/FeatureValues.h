// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/*
    Implementation of Feature Values
    All data types in onnxruntime\core\framework\data_types.cc should be implemented here
*/

#include "TensorBoolean.g.h"
#include "TensorFloat.g.h"
#include "TensorDouble.g.h"
#include "TensorInt8Bit.g.h"
#include "TensorUInt8Bit.g.h"
#include "TensorUInt16Bit.g.h"
#include "TensorInt16Bit.g.h"
#include "TensorUInt32Bit.g.h"
#include "TensorInt32Bit.g.h"
#include "TensorUInt64Bit.g.h"
#include "TensorInt64Bit.g.h"
#include "TensorFloat16Bit.g.h"
#include "TensorString.g.h"

#include "impl/MapBase.h"
#include "impl/SequenceBase.h"
#include "impl/TensorBase.h"

#include "ImageFeatureValue.h"

#define FREE_DIMENSION -1

// CREATE_TENSOR is used by data tensor types to implement common functionality
#define CREATE_TENSOR(type, element_type, element_view_type)                                       \
  namespace winrt::Windows::AI::MachineLearning::implementation {                                  \
  struct type : public WinML::TensorBase<                                                          \
                    element_type,                                                                  \
                    element_view_type,                                                             \
                    type,                                                                          \
                    I##type,                                                                       \
                    type##T<type,                                                                  \
                            ITensorNative,                                                         \
                            WinML::ILotusValueProviderPrivate>> {                                  \
    using Base =                                                                                   \
        TensorBase<                                                                                \
            element_type,                                                                          \
            element_view_type,                                                                     \
            type,                                                                                  \
            I##type,                                                                               \
            type##T<                                                                               \
                type,                                                                              \
                ITensorNative,                                                                     \
                WinML::ILotusValueProviderPrivate>>;                                               \
                                                                                                   \
    type() = default;                                                                              \
                                                                                                   \
    type(wfc::IIterable<int64_t> const& shape) : Base(shape){};                                    \
                                                                                                   \
    type(std::vector<int64_t> const& shape) : Base(shape){};                                       \
                                                                                                   \
    type(std::vector<int64_t> const& shape, ID3D12Resource* pResource) : Base(shape, pResource){}; \
  };                                                                                               \
  }                                                                                                \
  namespace winrt::Windows::AI::MachineLearning::factory_implementation {                          \
  struct type : type##T<type, implementation::type, ITensorStaticsNative> {                        \
    STDMETHOD(CreateFromD3D12Resource)                                                             \
    (ID3D12Resource * value, __int64* shape, int shapeSize, IUnknown** result) {                   \
      return implementation::type::CreateFromD3D12Resource(value, shape, shapeSize, result);       \
    }                                                                                              \
  };                                                                                               \
  }

CREATE_TENSOR(TensorBoolean, bool, bool)
CREATE_TENSOR(TensorFloat, float, float)
CREATE_TENSOR(TensorDouble, double, double)

// Currently, before the graph computation, we need to convert uint8 coming
// from application end to int8(ORT end) because winrt doesn't expose a signed 8-bit integer type,
// and after graph run, we need to convert it back.
CREATE_TENSOR(TensorInt8Bit, int8_t, uint8_t)
CREATE_TENSOR(TensorUInt8Bit, uint8_t, uint8_t)
CREATE_TENSOR(TensorUInt16Bit, uint16_t, uint16_t)
CREATE_TENSOR(TensorInt16Bit, int16_t, int16_t)
CREATE_TENSOR(TensorUInt32Bit, uint32_t, uint32_t)
CREATE_TENSOR(TensorInt32Bit, int32_t, int32_t)
CREATE_TENSOR(TensorUInt64Bit, uint64_t, uint64_t)
CREATE_TENSOR(TensorInt64Bit, int64_t, int64_t)
CREATE_TENSOR(TensorFloat16Bit, WinML::Half, float)

#pragma warning(push)
#pragma warning(disable : 4702) // Unreachable code (one of TensorBase's constructor unconditionally throws for
                                // std::string because it's not supported with D3D12 resources)
CREATE_TENSOR(TensorString, std::string, winrt::hstring)
#pragma warning(pop)

// CREATE_MAP is used by map types to implement common functionality
#define CREATE_MAP(type, key_type, value_type)                                                       \
  namespace winrt::Windows::AI::MachineLearning::implementation {                                    \
  struct type : public WinML::MapBase<type, key_type, value_type> {                                  \
    type(wfc::IMap<key_type, value_type> const& data) : MapBase<type, key_type, value_type>(data){}; \
  };                                                                                                 \
  }

CREATE_MAP(MapInt64BitToInt64Bit, int64_t, int64_t)
CREATE_MAP(MapInt64BitToFloat, int64_t, float)
CREATE_MAP(MapInt64BitToDouble, int64_t, double)
CREATE_MAP(MapInt64BitToString, int64_t, hstring)
CREATE_MAP(MapStringToInt64Bit, hstring, int64_t)
CREATE_MAP(MapStringToFloat, hstring, float)
CREATE_MAP(MapStringToDouble, hstring, double)
CREATE_MAP(MapStringToString, hstring, hstring)

// CREATE_SEQUENCE is used by sequence types to implement common functionality
#define CREATE_SEQUENCE(type, element_type)                                                    \
  namespace winrt::Windows::AI::MachineLearning::implementation {                              \
  struct type : public WinML::SequenceBase<type, element_type> {                               \
    type(wfc::IIterable<element_type> const& data) : SequenceBase<type, element_type>(data){}; \
  };                                                                                           \
  }

using AbiMapStringFloat = wfc::IMap<winrt::hstring, float>;
using AbiMapInt64BitFloat = wfc::IMap<int64_t, float>;

CREATE_SEQUENCE(SequenceMapStringFloat, AbiMapStringFloat)
CREATE_SEQUENCE(SequenceMapInt64BitFloat, AbiMapInt64BitFloat)

namespace Windows::AI::MachineLearning {

template <typename TValueType, typename TDataType>
inline winrt::Windows::AI::MachineLearning::ILearningModelFeatureValue CreateTensorValueFromInspectable(
    WinML::BindingType bindingType,
    const winrt::Windows::Foundation::IInspectable& inspectable,
    const winrt::Windows::AI::MachineLearning::ITensorFeatureDescriptor& descriptor) {
  namespace collections = winrt::Windows::Foundation::Collections;

  if (descriptor.TensorKind() == WinML::TensorKindFrom<TDataType>::Type) {
    if (auto vector = inspectable.try_as<collections::IVector<TDataType>>()) {
      return TValueType::CreateFromIterable(descriptor.Shape(), vector);
    }

    if (bindingType == Windows::AI::MachineLearning::BindingType::kInput) {
      // Feature inputs should be more permissive, and allow for views to be bound since they are read only
      if (auto vectorView = inspectable.try_as<collections::IVectorView<TDataType>>()) {
        return TValueType::CreateFromIterable(descriptor.Shape(), vectorView);
      }
    }
  }
  return nullptr;
}

template <>
inline winrt::Windows::AI::MachineLearning::ILearningModelFeatureValue CreateTensorValueFromInspectable<winrt::Windows::AI::MachineLearning::implementation::TensorInt8Bit, uint8_t>(
    WinML::BindingType bindingType,
    const winrt::Windows::Foundation::IInspectable& inspectable,
    const winrt::Windows::AI::MachineLearning::ITensorFeatureDescriptor& descriptor) {
  namespace abi = winrt::Windows::AI::MachineLearning;
  namespace impl = winrt::Windows::AI::MachineLearning::implementation;
  namespace collections = winrt::Windows::Foundation::Collections;

  if (descriptor.TensorKind() == abi::TensorKind::Int8) {
    if (auto vector = inspectable.try_as<collections::IVector<uint8_t>>()) {
      return impl::TensorInt8Bit::CreateFromIterable(descriptor.Shape(), vector);
    }

    if (bindingType == WinML::BindingType::kInput) {
      // Feature inputs should be more permissive, and allow for views to be bound since they are read only
      if (auto vectorView = inspectable.try_as<collections::IVectorView<uint8_t>>()) {
        return impl::TensorInt8Bit::CreateFromIterable(descriptor.Shape(), vectorView);
      }
    }
  }
  return nullptr;
}

template <>
inline winrt::Windows::AI::MachineLearning::ILearningModelFeatureValue CreateTensorValueFromInspectable<winrt::Windows::AI::MachineLearning::implementation::TensorFloat16Bit, float>(
    WinML::BindingType bindingType,
    const winrt::Windows::Foundation::IInspectable& inspectable,
    const winrt::Windows::AI::MachineLearning::ITensorFeatureDescriptor& descriptor) {
  namespace abi = winrt::Windows::AI::MachineLearning;
  namespace impl = winrt::Windows::AI::MachineLearning::implementation;
  namespace collections = winrt::Windows::Foundation::Collections;

  if (descriptor.TensorKind() == abi::TensorKind::Float16) {
    if (auto vector = inspectable.try_as<collections::IVector<float>>()) {
      return impl::TensorFloat16Bit::CreateFromIterable(descriptor.Shape(), vector);
    }

    if (bindingType == WinML::BindingType::kInput) {
      // Feature inputs should be more permissive, and allow for views to be bound since they are read only
      if (auto vectorView = inspectable.try_as<collections::IVectorView<float>>()) {
        return impl::TensorFloat16Bit::CreateFromIterable(descriptor.Shape(), vectorView);
      }
    }
  }
  return nullptr;
}

inline winrt::Windows::AI::MachineLearning::ILearningModelFeatureValue CreateFeatureValueFromInspectable(
    Windows::AI::MachineLearning::BindingType bindingType,
    const winrt::Windows::Foundation::IInspectable& inspectable,
    const winrt::Windows::AI::MachineLearning::ILearningModelFeatureDescriptor& descriptor) {
  using namespace winrt::Windows::AI::MachineLearning;
  using namespace winrt::Windows::Foundation::Collections;

  // Tensor and ImageFeatureValue types are passed in directly as feature values
  if (auto featureValue = inspectable.try_as<ILearningModelFeatureValue>()) {
    return featureValue;
  }

  if (auto videoFrames = inspectable.try_as<IVector<winrt::Windows::Media::VideoFrame>>()) {
    return (0 == videoFrames.Size()) ? nullptr : winrt::make<implementation::ImageFeatureValue>(videoFrames);
  }

  if (bindingType == Windows::AI::MachineLearning::BindingType::kInput) {
    // Allows to bind IVectorView<VideoFrame> as input.
    if (auto videoFrames = inspectable.try_as<IVectorView<winrt::Windows::Media::VideoFrame>>()) {
      return (0 == videoFrames.Size()) ? nullptr : winrt::make<implementation::ImageFeatureValue>(videoFrames);
    }
  }

  // ImageFeatureValues Types can be implicitly inferred from the VideoFrame object
  if (auto videoFrame = inspectable.try_as<winrt::Windows::Media::VideoFrame>()) {
    return winrt::make<implementation::ImageFeatureValue>(videoFrame);
  }

  // MapFeatureValues Types are implicitly inferred from the iinspectable object
  if (auto map = inspectable.try_as<IMap<winrt::hstring, float>>()) {
    return implementation::MapStringToFloat::Create(map);
  }
  if (auto map = inspectable.try_as<IMap<winrt::hstring, double>>()) {
    return implementation::MapStringToDouble::Create(map);
  }
  if (auto map = inspectable.try_as<IMap<winrt::hstring, int64_t>>()) {
    return implementation::MapStringToInt64Bit::Create(map);
  }
  if (auto map = inspectable.try_as<IMap<winrt::hstring, winrt::hstring>>()) {
    return implementation::MapStringToString::Create(map);
  }
  if (auto map = inspectable.try_as<IMap<int64_t, float>>()) {
    return implementation::MapInt64BitToFloat::Create(map);
  }
  if (auto map = inspectable.try_as<IMap<int64_t, double>>()) {
    return implementation::MapInt64BitToDouble::Create(map);
  }
  if (auto map = inspectable.try_as<IMap<int64_t, int64_t>>()) {
    return implementation::MapInt64BitToInt64Bit::Create(map);
  }
  if (auto map = inspectable.try_as<IMap<int64_t, winrt::hstring>>()) {
    return implementation::MapInt64BitToString::Create(map);
  }

  if (bindingType == Windows::AI::MachineLearning::BindingType::kInput) {
    // Feature inputs should be more permissive, and allow for views to be bound since they are read only
    if (auto map = inspectable.try_as<IMapView<winrt::hstring, float>>()) {
      return implementation::MapStringToFloat::Create(map);
    }
    if (auto map = inspectable.try_as<IMapView<winrt::hstring, double>>()) {
      return implementation::MapStringToDouble::Create(map);
    }
    if (auto map = inspectable.try_as<IMapView<winrt::hstring, int64_t>>()) {
      return implementation::MapStringToInt64Bit::Create(map);
    }
    if (auto map = inspectable.try_as<IMapView<winrt::hstring, winrt::hstring>>()) {
      return implementation::MapStringToString::Create(map);
    }
    if (auto map = inspectable.try_as<IMapView<int64_t, float>>()) {
      return implementation::MapInt64BitToFloat::Create(map);
    }
    if (auto map = inspectable.try_as<IMapView<int64_t, double>>()) {
      return implementation::MapInt64BitToDouble::Create(map);
    }
    if (auto map = inspectable.try_as<IMapView<int64_t, int64_t>>()) {
      return implementation::MapInt64BitToInt64Bit::Create(map);
    }
    if (auto map = inspectable.try_as<IMapView<int64_t, winrt::hstring>>()) {
      return implementation::MapInt64BitToString::Create(map);
    }
  }

  if (descriptor.Kind() == LearningModelFeatureKind::Sequence) {
    // SequenceFeatureValues Types are implicitly inferred from the iinspectable object
    if (auto sequence = inspectable.try_as<IVector<IMap<winrt::hstring, float>>>()) {
      return implementation::SequenceMapStringFloat::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<IVector<IMap<int64_t, float>>>()) {
      return implementation::SequenceMapInt64BitFloat::Create(sequence);
    }

    if (bindingType == Windows::AI::MachineLearning::BindingType::kInput) {
      // Feature inputs should be more permissive, and allow for views to be bound since they are read only
      if (auto sequence = inspectable.try_as<IVectorView<IMap<winrt::hstring, float>>>()) {
        return implementation::SequenceMapStringFloat::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<IVectorView<IMap<int64_t, float>>>()) {
        return implementation::SequenceMapInt64BitFloat::Create(sequence);
      }
    }
  } else if (descriptor.Kind() == LearningModelFeatureKind::Tensor) {
    using Value = winrt::Windows::AI::MachineLearning::ILearningModelFeatureValue;
    using Inspectable = winrt::Windows::Foundation::IInspectable;
    using Descriptor = winrt::Windows::AI::MachineLearning::ITensorFeatureDescriptor;
    using TensorCreator = std::function<Value()>;

    auto tensorDescriptor = descriptor.as<ITensorFeatureDescriptor>();
    std::vector<TensorCreator> creators =
        {
            // Vector and VectorViews of float16 and int8 collide with float and uint8 respectively.
            // They are omitted because of this ambiguity and are not constructible via raw winrt collections.
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorBoolean, bool>(bindingType, inspectable, tensorDescriptor); },
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorFloat, float>(bindingType, inspectable, tensorDescriptor); },
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorDouble, double>(bindingType, inspectable, tensorDescriptor); },
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorUInt8Bit, uint8_t>(bindingType, inspectable, tensorDescriptor); },
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorInt8Bit, uint8_t>(bindingType, inspectable, tensorDescriptor); },
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorUInt16Bit, uint16_t>(bindingType, inspectable, tensorDescriptor); },
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorInt16Bit, int16_t>(bindingType, inspectable, tensorDescriptor); },
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorUInt32Bit, uint32_t>(bindingType, inspectable, tensorDescriptor); },
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorInt32Bit, int32_t>(bindingType, inspectable, tensorDescriptor); },
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorUInt64Bit, uint64_t>(bindingType, inspectable, tensorDescriptor); },
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorInt64Bit, int64_t>(bindingType, inspectable, tensorDescriptor); },
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorFloat16Bit, float>(bindingType, inspectable, tensorDescriptor); },
            [&]() { return CreateTensorValueFromInspectable<implementation::TensorString, winrt::hstring>(bindingType, inspectable, tensorDescriptor); }};

    for (const auto& tensorCreator : creators) {
      if (auto createdTensor = tensorCreator()) {
        return createdTensor;
      }
    }
  }

  return nullptr;
}

}  // namespace Windows::AI::MachineLearning