#pragma once

namespace {
    template <winml::TensorKind T> struct TensorKindToPointerType { static_assert(true, "No TensorKind mapped for given type!"); };
    template <> struct TensorKindToPointerType<winml::TensorKind::UInt8> { typedef uint8_t Type; };
    template <> struct TensorKindToPointerType<winml::TensorKind::Int8> { typedef uint8_t Type; };
    template <> struct TensorKindToPointerType<winml::TensorKind::UInt16> { typedef uint16_t Type; };
    template <> struct TensorKindToPointerType<winml::TensorKind::Int16> { typedef int16_t Type; };
    template <> struct TensorKindToPointerType<winml::TensorKind::UInt32> { typedef uint32_t Type; };
    template <> struct TensorKindToPointerType<winml::TensorKind::Int32> { typedef int32_t Type; };
    template <> struct TensorKindToPointerType<winml::TensorKind::UInt64> { typedef uint64_t Type; };
    template <> struct TensorKindToPointerType<winml::TensorKind::Int64> { typedef int64_t Type; };
    template <> struct TensorKindToPointerType<winml::TensorKind::Boolean> { typedef boolean Type; };
    template <> struct TensorKindToPointerType<winml::TensorKind::Double> { typedef double Type; };
    template <> struct TensorKindToPointerType<winml::TensorKind::Float> { typedef float Type; };
    template <> struct TensorKindToPointerType<winml::TensorKind::Float16> { typedef float Type; };
    template <> struct TensorKindToPointerType<winml::TensorKind::String> { typedef winrt::hstring Type; };

    template <winml::TensorKind T> struct TensorKindToValue { static_assert(true, "No TensorKind mapped for given type!"); };
    template <> struct TensorKindToValue<winml::TensorKind::UInt8> { typedef winml::TensorUInt8Bit Type; };
    template <> struct TensorKindToValue<winml::TensorKind::Int8> { typedef winml::TensorInt8Bit Type; };
    template <> struct TensorKindToValue<winml::TensorKind::UInt16> { typedef winml::TensorUInt16Bit Type; };
    template <> struct TensorKindToValue<winml::TensorKind::Int16> { typedef winml::TensorInt16Bit Type; };
    template <> struct TensorKindToValue<winml::TensorKind::UInt32> { typedef winml::TensorUInt32Bit Type; };
    template <> struct TensorKindToValue<winml::TensorKind::Int32> { typedef winml::TensorInt32Bit Type; };
    template <> struct TensorKindToValue<winml::TensorKind::UInt64> { typedef winml::TensorUInt64Bit Type; };
    template <> struct TensorKindToValue<winml::TensorKind::Int64> { typedef winml::TensorInt64Bit Type; };
    template <> struct TensorKindToValue<winml::TensorKind::Boolean> { typedef winml::TensorBoolean Type; };
    template <> struct TensorKindToValue<winml::TensorKind::Double> { typedef winml::TensorDouble Type; };
    template <> struct TensorKindToValue<winml::TensorKind::Float> { typedef winml::TensorFloat Type; };
    template <> struct TensorKindToValue<winml::TensorKind::Float16> { typedef winml::TensorFloat16Bit Type; };
    template <> struct TensorKindToValue<winml::TensorKind::String> { typedef winml::TensorString Type; };
}