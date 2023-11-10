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

// CREATE_TENSOR is used by data tensor types to implement common functionality
#define CREATE_TENSOR(type, element_type, element_view_type)                                       \
  namespace WINMLP {                                                                               \
  struct type : public _winml::TensorBase<                                                         \
                  element_type,                                                                    \
                  element_view_type,                                                               \
                  type,                                                                            \
                  I##type,                                                                         \
                  type##T<type, ITensorNative, _winml::ILotusValueProviderPrivate>> {              \
    using Base = TensorBase<                                                                       \
      element_type,                                                                                \
      element_view_type,                                                                           \
      type,                                                                                        \
      I##type,                                                                                     \
      type##T<type, ITensorNative, _winml::ILotusValueProviderPrivate>>;                           \
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
  namespace WINML::factory_implementation {                                                        \
  struct type : type##T<type, winmlp::type, ITensorStaticsNative> {                                \
    STDMETHOD(CreateFromD3D12Resource)                                                             \
    (ID3D12Resource * value, __int64* shape, int shapeSize, IUnknown** result) {                   \
      return winmlp::type::CreateFromD3D12Resource(value, shape, shapeSize, result);               \
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
CREATE_TENSOR(TensorFloat16Bit, _winml::Half, float)

#pragma warning(push)
#pragma warning(disable : 4702) // Unreachable code (one of TensorBase's constructor unconditionally throws for \
                                // std::string because it's not supported with D3D12 resources)
CREATE_TENSOR(TensorString, std::string, winrt::hstring)
#pragma warning(pop)

// CREATE_MAP is used by map types to implement common functionality
#define CREATE_MAP(type, key_type, value_type)                                                       \
  namespace WINMLP {                                                                                 \
  struct type : public _winml::MapBase<type, key_type, value_type> {                                 \
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
#define CREATE_SEQUENCE(type, element_type, raw_type)                                                    \
  namespace WINMLP {                                                                                     \
  struct type : public _winml::SequenceBase<type, element_type, raw_type> {                              \
    type(wfc::IIterable<element_type> const& data) : SequenceBase<type, element_type, raw_type>(data){}; \
  };                                                                                                     \
  }

using AbiMapStringFloat = wfc::IMap<winrt::hstring, float>;
using AbiMapInt64BitFloat = wfc::IMap<int64_t, float>;

CREATE_SEQUENCE(SequenceMapStringFloat, AbiMapStringFloat, float)
CREATE_SEQUENCE(SequenceMapInt64BitFloat, AbiMapInt64BitFloat, float)
CREATE_SEQUENCE(SequenceTensorBoolean, winml::TensorBoolean, bool)
CREATE_SEQUENCE(SequenceTensorFloat, winml::TensorFloat, float)
CREATE_SEQUENCE(SequenceTensorDouble, winml::TensorDouble, double)
CREATE_SEQUENCE(SequenceTensorInt8Bit, winml::TensorInt8Bit, int8_t)
CREATE_SEQUENCE(SequenceTensorUInt8Bit, winml::TensorUInt8Bit, uint8_t)
CREATE_SEQUENCE(SequenceTensorUInt16Bit, winml::TensorUInt16Bit, uint16_t)
CREATE_SEQUENCE(SequenceTensorInt16Bit, winml::TensorInt16Bit, int16_t)
CREATE_SEQUENCE(SequenceTensorUInt32Bit, winml::TensorUInt32Bit, uint32_t)
CREATE_SEQUENCE(SequenceTensorInt32Bit, winml::TensorInt32Bit, int32_t)
CREATE_SEQUENCE(SequenceTensorUInt64Bit, winml::TensorUInt64Bit, uint64_t)
CREATE_SEQUENCE(SequenceTensorInt64Bit, winml::TensorInt64Bit, int64_t)
CREATE_SEQUENCE(SequenceTensorFloat16Bit, winml::TensorFloat16Bit, _winml::Half)
CREATE_SEQUENCE(SequenceTensorString, winml::TensorString, std::string)

namespace _winml {

template <typename TValueType, typename TDataType>
inline winml::ILearningModelFeatureValue CreateTensorValueFromInspectable(
  _winml::BindingType bindingType,
  const wf::IInspectable& inspectable,
  const winml::ITensorFeatureDescriptor& descriptor
) {
  if (descriptor.TensorKind() == _winml::TensorKindFrom<TDataType>::Type) {
    if (auto vector = inspectable.try_as<wfc::IVector<TDataType>>()) {
      return TValueType::CreateFromIterable(descriptor.Shape(), vector);
    }

    if (bindingType == _winml::BindingType::kInput) {
      // Feature inputs should be more permissive, and allow for views to be bound since they are read only
      if (auto vectorView = inspectable.try_as<wfc::IVectorView<TDataType>>()) {
        return TValueType::CreateFromIterable(descriptor.Shape(), vectorView);
      }
    }
  }
  return nullptr;
}

template <>
inline winml::ILearningModelFeatureValue CreateTensorValueFromInspectable<winmlp::TensorInt8Bit, uint8_t>(
  _winml::BindingType bindingType,
  const wf::IInspectable& inspectable,
  const winml::ITensorFeatureDescriptor& descriptor
) {
  if (descriptor.TensorKind() == winml::TensorKind::Int8) {
    if (auto vector = inspectable.try_as<wfc::IVector<uint8_t>>()) {
      return winmlp::TensorInt8Bit::CreateFromIterable(descriptor.Shape(), vector);
    }

    if (bindingType == _winml::BindingType::kInput) {
      // Feature inputs should be more permissive, and allow for views to be bound since they are read only
      if (auto vectorView = inspectable.try_as<wfc::IVectorView<uint8_t>>()) {
        return winmlp::TensorInt8Bit::CreateFromIterable(descriptor.Shape(), vectorView);
      }
    }
  }
  return nullptr;
}

template <>
inline winml::ILearningModelFeatureValue CreateTensorValueFromInspectable<winmlp::TensorFloat16Bit, float>(
  _winml::BindingType bindingType,
  const wf::IInspectable& inspectable,
  const winml::ITensorFeatureDescriptor& descriptor
) {
  if (descriptor.TensorKind() == winml::TensorKind::Float16) {
    if (auto vector = inspectable.try_as<wfc::IVector<float>>()) {
      return winmlp::TensorFloat16Bit::CreateFromIterable(descriptor.Shape(), vector);
    }

    if (bindingType == _winml::BindingType::kInput) {
      // Feature inputs should be more permissive, and allow for views to be bound since they are read only
      if (auto vectorView = inspectable.try_as<wfc::IVectorView<float>>()) {
        return winmlp::TensorFloat16Bit::CreateFromIterable(descriptor.Shape(), vectorView);
      }
    }
  }
  return nullptr;
}

inline winml::ILearningModelFeatureValue CreateFeatureValueFromInspectable(
  _winml::BindingType bindingType,
  const wf::IInspectable& inspectable,
  const winml::ILearningModelFeatureDescriptor& descriptor
) {
  // Tensor and ImageFeatureValue types are passed in directly as feature values
  if (auto featureValue = inspectable.try_as<winml::ILearningModelFeatureValue>()) {
    return featureValue;
  }

  if (auto videoFrames = inspectable.try_as<wfc::IVector<wm::VideoFrame>>()) {
    return (0 == videoFrames.Size()) ? nullptr : winrt::make<winmlp::ImageFeatureValue>(videoFrames);
  }

  if (bindingType == _winml::BindingType::kInput) {
    // Allows to bind IVectorView<VideoFrame> as input.
    if (auto videoFrames = inspectable.try_as<wfc::IVectorView<wm::VideoFrame>>()) {
      return (0 == videoFrames.Size()) ? nullptr : winrt::make<winmlp::ImageFeatureValue>(videoFrames);
    }
  }

  // ImageFeatureValues Types can be implicitly inferred from the VideoFrame object
  if (auto videoFrame = inspectable.try_as<wm::VideoFrame>()) {
    return winrt::make<winmlp::ImageFeatureValue>(videoFrame);
  }

  // MapFeatureValues Types are implicitly inferred from the iinspectable object
  if (auto map = inspectable.try_as<wfc::IMap<winrt::hstring, float>>()) {
    return winmlp::MapStringToFloat::Create(map);
  }
  if (auto map = inspectable.try_as<wfc::IMap<winrt::hstring, double>>()) {
    return winmlp::MapStringToDouble::Create(map);
  }
  if (auto map = inspectable.try_as<wfc::IMap<winrt::hstring, int64_t>>()) {
    return winmlp::MapStringToInt64Bit::Create(map);
  }
  if (auto map = inspectable.try_as<wfc::IMap<winrt::hstring, winrt::hstring>>()) {
    return winmlp::MapStringToString::Create(map);
  }
  if (auto map = inspectable.try_as<wfc::IMap<int64_t, float>>()) {
    return winmlp::MapInt64BitToFloat::Create(map);
  }
  if (auto map = inspectable.try_as<wfc::IMap<int64_t, double>>()) {
    return winmlp::MapInt64BitToDouble::Create(map);
  }
  if (auto map = inspectable.try_as<wfc::IMap<int64_t, int64_t>>()) {
    return winmlp::MapInt64BitToInt64Bit::Create(map);
  }
  if (auto map = inspectable.try_as<wfc::IMap<int64_t, winrt::hstring>>()) {
    return winmlp::MapInt64BitToString::Create(map);
  }

  if (bindingType == _winml::BindingType::kInput) {
    // Feature inputs should be more permissive, and allow for views to be bound since they are read only
    if (auto map = inspectable.try_as<wfc::IMapView<winrt::hstring, float>>()) {
      return winmlp::MapStringToFloat::Create(map);
    }
    if (auto map = inspectable.try_as<wfc::IMapView<winrt::hstring, double>>()) {
      return winmlp::MapStringToDouble::Create(map);
    }
    if (auto map = inspectable.try_as<wfc::IMapView<winrt::hstring, int64_t>>()) {
      return winmlp::MapStringToInt64Bit::Create(map);
    }
    if (auto map = inspectable.try_as<wfc::IMapView<winrt::hstring, winrt::hstring>>()) {
      return winmlp::MapStringToString::Create(map);
    }
    if (auto map = inspectable.try_as<wfc::IMapView<int64_t, float>>()) {
      return winmlp::MapInt64BitToFloat::Create(map);
    }
    if (auto map = inspectable.try_as<wfc::IMapView<int64_t, double>>()) {
      return winmlp::MapInt64BitToDouble::Create(map);
    }
    if (auto map = inspectable.try_as<wfc::IMapView<int64_t, int64_t>>()) {
      return winmlp::MapInt64BitToInt64Bit::Create(map);
    }
    if (auto map = inspectable.try_as<wfc::IMapView<int64_t, winrt::hstring>>()) {
      return winmlp::MapInt64BitToString::Create(map);
    }
  }

  if (descriptor.Kind() == winml::LearningModelFeatureKind::Sequence) {
    // SequenceFeatureValues Types are implicitly inferred from the iinspectable object
    if (auto sequence = inspectable.try_as<wfc::IVector<wfc::IMap<winrt::hstring, float>>>()) {
      return winmlp::SequenceMapStringFloat::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<wfc::IMap<int64_t, float>>>()) {
      return winmlp::SequenceMapInt64BitFloat::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorFloat>>()) {
      return winmlp::SequenceTensorFloat::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorBoolean>>()) {
      return winmlp::SequenceTensorBoolean::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorDouble>>()) {
      return winmlp::SequenceTensorDouble::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorInt8Bit>>()) {
      return winmlp::SequenceTensorInt8Bit::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorUInt8Bit>>()) {
      return winmlp::SequenceTensorUInt8Bit::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorUInt16Bit>>()) {
      return winmlp::SequenceTensorUInt16Bit::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorInt16Bit>>()) {
      return winmlp::SequenceTensorInt16Bit::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorUInt32Bit>>()) {
      return winmlp::SequenceTensorUInt32Bit::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorInt32Bit>>()) {
      return winmlp::SequenceTensorInt32Bit::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorUInt64Bit>>()) {
      return winmlp::SequenceTensorUInt64Bit::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorInt64Bit>>()) {
      return winmlp::SequenceTensorInt64Bit::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorFloat16Bit>>()) {
      return winmlp::SequenceTensorFloat16Bit::Create(sequence);
    }
    if (auto sequence = inspectable.try_as<wfc::IVector<winml::TensorString>>()) {
      return winmlp::SequenceTensorString::Create(sequence);
    }

    if (bindingType == _winml::BindingType::kInput) {
      // Feature inputs should be more permissive, and allow for views to be bound since they are read only
      if (auto sequence = inspectable.try_as<wfc::IVectorView<wfc::IMap<winrt::hstring, float>>>()) {
        return winmlp::SequenceMapStringFloat::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<wfc::IMap<int64_t, float>>>()) {
        return winmlp::SequenceMapInt64BitFloat::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorFloat>>()) {
        return winmlp::SequenceTensorFloat::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorBoolean>>()) {
        return winmlp::SequenceTensorBoolean::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorDouble>>()) {
        return winmlp::SequenceTensorDouble::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorInt8Bit>>()) {
        return winmlp::SequenceTensorInt8Bit::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorUInt8Bit>>()) {
        return winmlp::SequenceTensorUInt8Bit::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorUInt16Bit>>()) {
        return winmlp::SequenceTensorUInt16Bit::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorInt16Bit>>()) {
        return winmlp::SequenceTensorInt16Bit::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorUInt32Bit>>()) {
        return winmlp::SequenceTensorUInt32Bit::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorInt32Bit>>()) {
        return winmlp::SequenceTensorInt32Bit::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorUInt64Bit>>()) {
        return winmlp::SequenceTensorUInt64Bit::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorInt64Bit>>()) {
        return winmlp::SequenceTensorInt64Bit::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorFloat16Bit>>()) {
        return winmlp::SequenceTensorFloat16Bit::Create(sequence);
      }
      if (auto sequence = inspectable.try_as<wfc::IVectorView<winml::TensorString>>()) {
        return winmlp::SequenceTensorString::Create(sequence);
      }
    }
  } else if (descriptor.Kind() == winml::LearningModelFeatureKind::Tensor) {
    auto tensorDescriptor = descriptor.as<winml::ITensorFeatureDescriptor>();

    // Vector of IBuffer Input should be copied into the appropriate Tensor
    if (auto buffers = inspectable.try_as<wfc::IIterable<wss::IBuffer>>()) {
      if (tensorDescriptor.TensorKind() == winml::TensorKind::Boolean) {
        return winmlp::TensorBoolean::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
      if (tensorDescriptor.TensorKind() == winml::TensorKind::Float) {
        return winmlp::TensorFloat::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
      if (tensorDescriptor.TensorKind() == winml::TensorKind::Double) {
        return winmlp::TensorDouble::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
      if (tensorDescriptor.TensorKind() == winml::TensorKind::Float16) {
        return winmlp::TensorFloat16Bit::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
      if (tensorDescriptor.TensorKind() == winml::TensorKind::UInt8) {
        return winmlp::TensorUInt8Bit::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
      if (tensorDescriptor.TensorKind() == winml::TensorKind::Int8) {
        return winmlp::TensorInt8Bit::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
      if (tensorDescriptor.TensorKind() == winml::TensorKind::UInt16) {
        return winmlp::TensorUInt16Bit::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
      if (tensorDescriptor.TensorKind() == winml::TensorKind::Int16) {
        return winmlp::TensorInt16Bit::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
      if (tensorDescriptor.TensorKind() == winml::TensorKind::UInt32) {
        return winmlp::TensorUInt32Bit::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
      if (tensorDescriptor.TensorKind() == winml::TensorKind::Int32) {
        return winmlp::TensorInt32Bit::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
      if (tensorDescriptor.TensorKind() == winml::TensorKind::UInt64) {
        return winmlp::TensorUInt64Bit::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
      if (tensorDescriptor.TensorKind() == winml::TensorKind::Int64) {
        return winmlp::TensorInt64Bit::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
      if (tensorDescriptor.TensorKind() == winml::TensorKind::Float16) {
        return winmlp::TensorFloat16Bit::CreateFromBatchedBuffers(tensorDescriptor.Shape(), buffers);
      }
    }

    using TensorCreator = winml::ILearningModelFeatureValue (*)(
      BindingType, const wf::IInspectable& inspectable, const winml::ITensorFeatureDescriptor& descriptor
    );
    constexpr std::array<TensorCreator, 13> creators = {
            // Vector and VectorViews of float16 and int8 collide with float and uint8 respectively.
            // They are omitted because of this ambiguity and are not constructible via raw winrt collections.
      CreateTensorValueFromInspectable<winmlp::TensorBoolean, bool>,
      CreateTensorValueFromInspectable<winmlp::TensorFloat, float>,
      CreateTensorValueFromInspectable<winmlp::TensorDouble, double>,
      CreateTensorValueFromInspectable<winmlp::TensorUInt8Bit, uint8_t>,
      CreateTensorValueFromInspectable<winmlp::TensorInt8Bit, uint8_t>,
      CreateTensorValueFromInspectable<winmlp::TensorUInt16Bit, uint16_t>,
      CreateTensorValueFromInspectable<winmlp::TensorInt16Bit, int16_t>,
      CreateTensorValueFromInspectable<winmlp::TensorUInt32Bit, uint32_t>,
      CreateTensorValueFromInspectable<winmlp::TensorInt32Bit, int32_t>,
      CreateTensorValueFromInspectable<winmlp::TensorUInt64Bit, uint64_t>,
      CreateTensorValueFromInspectable<winmlp::TensorInt64Bit, int64_t>,
      CreateTensorValueFromInspectable<winmlp::TensorFloat16Bit, float>,
      CreateTensorValueFromInspectable<winmlp::TensorString, winrt::hstring>};

    for (const auto& tensorCreator : creators) {
      if (auto createdTensor = tensorCreator(bindingType, inspectable, tensorDescriptor)) {
        return createdTensor;
      }
    }
  }

  return nullptr;
}

}  // namespace _winml
