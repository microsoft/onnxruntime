// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace _winml {

// We need to define our own type for Half since DirectX::PackedVector::Half resolves to uint16_t per its typedef declaration.
// Templates require an actual type name to resolve correctly.
struct Half {
  DirectX::PackedVector::HALF value;
};

template <typename T>
struct TensorKindFrom {};
template <>
struct TensorKindFrom<winml::ITensorInt8Bit> {
  static const winml::TensorKind Type = winml::TensorKind::Int8;
};
template <>
struct TensorKindFrom<winml::ITensorUInt8Bit> {
  static const winml::TensorKind Type = winml::TensorKind::UInt8;
};
template <>
struct TensorKindFrom<winml::ITensorUInt16Bit> {
  static const winml::TensorKind Type = winml::TensorKind::UInt16;
};
template <>
struct TensorKindFrom<winml::ITensorInt16Bit> {
  static const winml::TensorKind Type = winml::TensorKind::Int16;
};
template <>
struct TensorKindFrom<winml::ITensorUInt32Bit> {
  static const winml::TensorKind Type = winml::TensorKind::UInt32;
};
template <>
struct TensorKindFrom<winml::ITensorInt32Bit> {
  static const winml::TensorKind Type = winml::TensorKind::Int32;
};
template <>
struct TensorKindFrom<winml::ITensorUInt64Bit> {
  static const winml::TensorKind Type = winml::TensorKind::UInt64;
};
template <>
struct TensorKindFrom<winml::ITensorInt64Bit> {
  static const winml::TensorKind Type = winml::TensorKind::Int64;
};
template <>
struct TensorKindFrom<winml::ITensorBoolean> {
  static const winml::TensorKind Type = winml::TensorKind::Boolean;
};
template <>
struct TensorKindFrom<winml::ITensorDouble> {
  static const winml::TensorKind Type = winml::TensorKind::Double;
};
template <>
struct TensorKindFrom<winml::ITensorFloat> {
  static const winml::TensorKind Type = winml::TensorKind::Float;
};
template <>
struct TensorKindFrom<winml::ITensorFloat16Bit> {
  static const winml::TensorKind Type = winml::TensorKind::Float16;
};
template <>
struct TensorKindFrom<winml::ITensorString> {
  static const winml::TensorKind Type = winml::TensorKind::String;
};

template <>
struct TensorKindFrom<int8_t> {
  static const winml::TensorKind Type = winml::TensorKind::Int8;
};
template <>
struct TensorKindFrom<uint8_t> {
  static const winml::TensorKind Type = winml::TensorKind::UInt8;
};
template <>
struct TensorKindFrom<uint16_t> {
  static const winml::TensorKind Type = winml::TensorKind::UInt16;
};
template <>
struct TensorKindFrom<int16_t> {
  static const winml::TensorKind Type = winml::TensorKind::Int16;
};
template <>
struct TensorKindFrom<uint32_t> {
  static const winml::TensorKind Type = winml::TensorKind::UInt32;
};
template <>
struct TensorKindFrom<int32_t> {
  static const winml::TensorKind Type = winml::TensorKind::Int32;
};
template <>
struct TensorKindFrom<uint64_t> {
  static const winml::TensorKind Type = winml::TensorKind::UInt64;
};
template <>
struct TensorKindFrom<int64_t> {
  static const winml::TensorKind Type = winml::TensorKind::Int64;
};
template <>
struct TensorKindFrom<bool> {
  static const winml::TensorKind Type = winml::TensorKind::Boolean;
};
template <>
struct TensorKindFrom<double> {
  static const winml::TensorKind Type = winml::TensorKind::Double;
};
template <>
struct TensorKindFrom<float> {
  static const winml::TensorKind Type = winml::TensorKind::Float;
};
template <>
struct TensorKindFrom<winrt::hstring> {
  static const winml::TensorKind Type = winml::TensorKind::String;
};
template <>
struct TensorKindFrom<std::string> {
  static const winml::TensorKind Type = winml::TensorKind::String;
};
template <>
struct TensorKindFrom<Half> {
  static const winml::TensorKind Type = winml::TensorKind::Float16;
};

template <typename T>
struct TensorFeatureDescriptorFrom {
  static winml::ILearningModelFeatureDescriptor CreateAnonymous(std::vector<int64_t> shape) {
    return winrt::make<winmlp::TensorFeatureDescriptor>(
      nullptr /* set to null as values are name-less */,
      nullptr /* set to null as values are description-less */,
      TensorKindFrom<T>::Type,
      shape,
      false /* set to false as values dont have required annotations */,
      false /* set to false as this is not a tensor of unsupported metadata */
    );
  }
};

}  // namespace _winml
