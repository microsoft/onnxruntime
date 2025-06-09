// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>

#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/type_list.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/data_types.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math_cpuonly.h"

#include "Eigen/src/Core/arch/Default/BFloat16.h"
#include "Eigen/src/Core/arch/Default/Half.h"

#include "core/mlas/inc/mlas.h"
#include "core/common/cpuid_info.h"

namespace onnxruntime {

namespace op_kernel_type_control {
// we're using one set of types for all opsets of Cast
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Input, 0,
    element_type_lists::AllIRv10);

ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Input, 0,
    bool, int32_t, int64_t);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Output, 0,
    element_type_lists::AllIRv10);

ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Output, 0,
    bool, int32_t, int64_t);
}  // namespace op_kernel_type_control

namespace {
using EnabledSrcTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                       Cast, Input, 0);
using EnabledDstTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                       Cast, Output, 0);

template <typename T>
using IsOrtFloat16Type = boost::mp11::mp_contains<TypeList<BFloat16, MLFloat16>, T>;

#if !defined(DISABLE_FLOAT8_TYPES)
template <typename T>
using IsOrtFloat8Type = boost::mp11::mp_contains<element_type_lists::AllFloat8, T>;
#endif

template <typename T>
struct IsStandardIntegerType {
  static constexpr bool value =
      std::is_same<T, int8_t>::value ||
      std::is_same<T, uint8_t>::value ||
      std::is_same<T, int16_t>::value ||
      std::is_same<T, uint16_t>::value ||
      std::is_same<T, int32_t>::value ||
      std::is_same<T, uint32_t>::value ||
      std::is_same<T, int64_t>::value ||
      std::is_same<T, uint64_t>::value;
};

template <typename T>
struct IsStandardFloatType {
  static constexpr bool value =
      std::is_same<T, float>::value ||
      std::is_same<T, double>::value;
};

// string cast helpers
// Note: when C++17 is available, use <charconv> functions

// handle floating point output separately
template <typename SrcType>
typename std::enable_if<std::is_floating_point<SrcType>::value, void>::type
CastToString(const SrcType& input, std::string& output) {
  static_assert(sizeof(SrcType) <= sizeof(double),
                "largest supported floating point type is double");
  if (std::isnan(input)) {
    output = "NaN";
  } else if (std::isinf(input)) {
    if (input < std::numeric_limits<SrcType>::lowest()) {
      output = "-INF";
    } else {
      output = "INF";
    }
  } else {
    // set precision to 8 to match numpy default behavior
    constexpr const char* format = "%.8g";
    const double value = static_cast<double>(input);

    char static_buffer[256];
    std::unique_ptr<char[]> dynamic_buffer{};

    gsl::span<char> buffer_span = gsl::make_span(static_buffer);

    auto snprintf_result = std::snprintf(buffer_span.data(), buffer_span.size(), format, value);
    ORT_ENFORCE(snprintf_result > 0, "snprintf() failed with return value: ", snprintf_result);

    // include trailing '\0'
    const size_t required_buffer_size = gsl::narrow_cast<size_t>(snprintf_result) + 1;

    if (required_buffer_size > buffer_span.size()) {
      // didn't get it all, allocate a bigger buffer and retry
      dynamic_buffer = std::make_unique<char[]>(required_buffer_size);
      buffer_span = gsl::make_span(dynamic_buffer.get(), required_buffer_size);
      snprintf_result = std::snprintf(buffer_span.data(), buffer_span.size(), format, value);
      ORT_ENFORCE(
          snprintf_result > 0 &&
              gsl::narrow_cast<size_t>(snprintf_result) == buffer_span.size() - 1,
          "Failed to write value with snprintf().");
    }

    output.assign(buffer_span.data(), required_buffer_size - 1);
  }
}

template <typename SrcType>
typename std::enable_if<std::is_integral<SrcType>::value, void>::type
CastToString(const SrcType& input, std::string& output) {
  output = std::to_string(input);
}

template <typename SrcType>
#if !defined(DISABLE_FLOAT8_TYPES)
typename std::enable_if<IsOrtFloat16Type<SrcType>::value || IsOrtFloat8Type<SrcType>::value, void>::type
#else
typename std::enable_if<IsOrtFloat16Type<SrcType>::value>::type
#endif
CastToString(const SrcType& input, std::string& output) {
  CastToString(static_cast<float>(input), output);
}

inline void CastToString(Int4x2 value, std::string& out) {
  // Int4x2 contains two 4-bit signed integers
  // Show both values as [first,second]
  auto val0 = value.GetElem(0);  // First 4-bit value
  auto val1 = value.GetElem(1);  // Second 4-bit value
  out = "[" + std::to_string(static_cast<int>(val0)) + "," + std::to_string(static_cast<int>(val1)) + "]";
}

inline void CastToString(UInt4x2 value, std::string& out) {
  // UInt4x2 contains two 4-bit unsigned integers
  auto val0 = value.GetElem(0);  // First 4-bit value
  auto val1 = value.GetElem(1);  // Second 4-bit value
  out = "[" + std::to_string(static_cast<unsigned>(val0)) + "," + std::to_string(static_cast<unsigned>(val1)) + "]";
}

template <typename DstType>
typename std::enable_if<std::is_floating_point<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  static_assert(sizeof(DstType) <= sizeof(double),
                "largest supported floating point type is double");
  output = gsl::narrow_cast<DstType>(std::stod(input));
}

template <typename DstType>
typename std::enable_if<std::is_integral<DstType>::value && std::is_unsigned<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  static_assert(sizeof(DstType) <= sizeof(unsigned long long),
                "largest supported unsigned integral type is unsigned long long");
  output = gsl::narrow_cast<DstType>(std::stoull(input));
}

template <typename DstType>
typename std::enable_if<std::is_integral<DstType>::value && std::is_signed<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  static_assert(sizeof(DstType) <= sizeof(long long),
                "largest supported signed integral type is long long");
  output = gsl::narrow_cast<DstType>(std::stoll(input));
}

inline void CastFromString(const std::string& in, Int4x2& out) {
  // Parse string format: "[-3,7]" or "-3,7" or just "-3" (single value)
  std::string trimmed = in;

  // Remove brackets if present
  if (!trimmed.empty() && trimmed.front() == '[') {
    trimmed = trimmed.substr(1);
  }
  if (!trimmed.empty() && trimmed.back() == ']') {
    trimmed = trimmed.substr(0, trimmed.length() - 1);
  }

  // Find comma separator
  size_t comma_pos = trimmed.find(',');
  int8_t val0 = 0, val1 = 0;
  if (comma_pos != std::string::npos) {
    // Two values: "val0,val1"
    std::string val0_str = trimmed.substr(0, comma_pos);
    std::string val1_str = trimmed.substr(comma_pos + 1);

    val0 = static_cast<int8_t>(std::clamp(std::stoi(val0_str), -8, 7));
    val1 = static_cast<int8_t>(std::clamp(std::stoi(val1_str), -8, 7));
  } else {
    // Single value - use for both elements
    val0 = val1 = static_cast<int8_t>(std::clamp(std::stoi(trimmed), -8, 7));
  }

  out = Int4x2(val0, val1);
}

inline void CastFromString(const std::string& in, UInt4x2& out) {
  // Parse string format: "[5,12]" or "5,12" or just "5" (single value)
  std::string trimmed = in;

  // Remove brackets if present
  if (!trimmed.empty() && trimmed.front() == '[') {
    trimmed = trimmed.substr(1);
  }
  if (!trimmed.empty() && trimmed.back() == ']') {
    trimmed = trimmed.substr(0, trimmed.length() - 1);
  }

  // Find comma separator
  size_t comma_pos = trimmed.find(',');
  uint8_t val0 = 0, val1 = 0;
  if (comma_pos != std::string::npos) {
    // Two values: "val0,val1"
    std::string val0_str = trimmed.substr(0, comma_pos);
    std::string val1_str = trimmed.substr(comma_pos + 1);

    val0 = static_cast<uint8_t>(std::clamp(std::stoi(val0_str), 0, 15));
    val1 = static_cast<uint8_t>(std::clamp(std::stoi(val1_str), 0, 15));
  } else {
    // Single value - use for both elements
    val0 = val1 = static_cast<uint8_t>(std::clamp(std::stoi(trimmed), 0, 15));
  }

  out = UInt4x2(val0, val1);
}

template <typename DstType>
#if !defined(DISABLE_FLOAT8_TYPES)
typename std::enable_if<IsOrtFloat16Type<DstType>::value || IsOrtFloat8Type<DstType>::value, void>::type
#else
typename std::enable_if<IsOrtFloat16Type<DstType>::value, void>::type
#endif
CastFromString(const std::string& input, DstType& output) {
  float intermediate;
  CastFromString(input, intermediate);
  output = DstType(intermediate);
}

// type that is usable with Eigen cast
template <typename T>
struct EigenCastType {
  using type = T;
};

// ORT float16 types don't support Eigen cast, so map them to Eigen ones

template <>
struct EigenCastType<MLFloat16> {
  using type = Eigen::half;
};

template <>
struct EigenCastType<BFloat16> {
  using type = Eigen::bfloat16;
};

// Helper struct for converting from Int4x2/UInt4x2 elements to any destination type
namespace {
template <typename DstType>
struct Int4ElementConverter {
  static DstType Convert(int8_t val) {
    // Default implementation for most numeric types
    return static_cast<DstType>(val);
  }
};

template <>
struct Int4ElementConverter<MLFloat16> {
  static MLFloat16 Convert(int8_t val) {
    return MLFloat16(static_cast<float>(val));
  }
};

template <>
struct Int4ElementConverter<BFloat16> {
  static BFloat16 Convert(int8_t val) {
    return BFloat16(static_cast<float>(val));
  }
};

template <>
struct Int4ElementConverter<std::string> {
  static std::string Convert(int8_t val) {
    return std::to_string(static_cast<int>(val));
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

template <>
struct Int4ElementConverter<Float8E4M3FN> {
  static Float8E4M3FN Convert(int8_t val) {
    return Float8E4M3FN(static_cast<float>(val), true);
  }
};

template <>
struct Int4ElementConverter<Float8E4M3FNUZ> {
  static Float8E4M3FNUZ Convert(int8_t val) {
    return Float8E4M3FNUZ(static_cast<float>(val), true);
  }
};

template <>
struct Int4ElementConverter<Float8E5M2> {
  static Float8E5M2 Convert(int8_t val) {
    return Float8E5M2(static_cast<float>(val), true);
  }
};

template <>
struct Int4ElementConverter<Float8E5M2FNUZ> {
  static Float8E5M2FNUZ Convert(int8_t val) {
    return Float8E5M2FNUZ(static_cast<float>(val), true);
  }
};
#endif

// Helper struct for converting from any type to Int4/UInt4 elements
template <typename SrcType>
struct ToInt4ElementConverter {
  // Default implementation for most numeric types
  static int8_t ConvertToInt4(const SrcType& val) {
    int8_t result = static_cast<int8_t>(val);
    // Clamp to int4 range (-8 to 7)
    return std::clamp(result, static_cast<int8_t>(-8), static_cast<int8_t>(7));
  }

  static uint8_t ConvertToUInt4(const SrcType& val) {
    uint8_t result = static_cast<uint8_t>(val);
    // Clamp to uint4 range (0 to 15)
    return std::min(result, static_cast<uint8_t>(15));
  }
};

template <>
struct ToInt4ElementConverter<float> {
  static int8_t ConvertToInt4(const float& val) {
    int8_t result = static_cast<int8_t>(std::roundf(val));
    return std::clamp(result, static_cast<int8_t>(-8), static_cast<int8_t>(7));
  }

  static uint8_t ConvertToUInt4(const float& val) {
    uint8_t result = static_cast<uint8_t>(std::max(0.0f, std::roundf(val)));
    return std::min(result, static_cast<uint8_t>(15));
  }
};

template <>
struct ToInt4ElementConverter<double> {
  static int8_t ConvertToInt4(const double& val) {
    int8_t result = static_cast<int8_t>(std::round(val));
    return std::clamp(result, static_cast<int8_t>(-8), static_cast<int8_t>(7));
  }

  static uint8_t ConvertToUInt4(const double& val) {
    uint8_t result = static_cast<uint8_t>(std::max(0.0, std::round(val)));
    return std::min(result, static_cast<uint8_t>(15));
  }
};

template <>
struct ToInt4ElementConverter<BFloat16> {
  static int8_t ConvertToInt4(const BFloat16& val) {
    return ToInt4ElementConverter<float>::ConvertToInt4(static_cast<float>(val));
  }

  static uint8_t ConvertToUInt4(const BFloat16& val) {
    return ToInt4ElementConverter<float>::ConvertToUInt4(static_cast<float>(val));
  }
};

template <>
struct ToInt4ElementConverter<MLFloat16> {
  static int8_t ConvertToInt4(const MLFloat16& val) {
    return ToInt4ElementConverter<float>::ConvertToInt4(static_cast<float>(val));
  }

  static uint8_t ConvertToUInt4(const MLFloat16& val) {
    return ToInt4ElementConverter<float>::ConvertToUInt4(static_cast<float>(val));
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

template <>
struct ToInt4ElementConverter<Float8E4M3FN> {
  static int8_t ConvertToInt4(const Float8E4M3FN& val) {
    return ToInt4ElementConverter<float>::ConvertToInt4(static_cast<float>(val));
  }

  static uint8_t ConvertToUInt4(const Float8E4M3FN& val) {
    return ToInt4ElementConverter<float>::ConvertToUInt4(static_cast<float>(val));
  }
};

template <>
struct ToInt4ElementConverter<Float8E4M3FNUZ> {
  static int8_t ConvertToInt4(const Float8E4M3FNUZ& val) {
    return ToInt4ElementConverter<float>::ConvertToInt4(static_cast<float>(val));
  }

  static uint8_t ConvertToUInt4(const Float8E4M3FNUZ& val) {
    return ToInt4ElementConverter<float>::ConvertToUInt4(static_cast<float>(val));
  }
};

template <>
struct ToInt4ElementConverter<Float8E5M2> {
  static int8_t ConvertToInt4(const Float8E5M2& val) {
    return ToInt4ElementConverter<float>::ConvertToInt4(static_cast<float>(val));
  }

  static uint8_t ConvertToUInt4(const Float8E5M2& val) {
    return ToInt4ElementConverter<float>::ConvertToUInt4(static_cast<float>(val));
  }
};

template <>
struct ToInt4ElementConverter<Float8E5M2FNUZ> {
  static int8_t ConvertToInt4(const Float8E5M2FNUZ& val) {
    return ToInt4ElementConverter<float>::ConvertToInt4(static_cast<float>(val));
  }

  static uint8_t ConvertToUInt4(const Float8E5M2FNUZ& val) {
    return ToInt4ElementConverter<float>::ConvertToUInt4(static_cast<float>(val));
  }
};

#endif

template <>
struct ToInt4ElementConverter<std::string> {
  static int8_t ConvertToInt4(const std::string& val) {
    int result;
    try {
      result = std::stoi(val);
    } catch (...) {
      result = 0;
    }
    return std::clamp(result, -8, 7);
  }

  static uint8_t ConvertToUInt4(const std::string& val) {
    unsigned int result;
    try {
      result = std::stoul(val);
    } catch (...) {
      result = 0;
    }
    return std::min(result, 15u);
  }
};

}  // anonymous namespace

// generic tensor X -> Y
template <typename SrcType, typename DstType, typename Enable = void>
struct TensorCaster {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    using SrcEigenCastType = typename EigenCastType<SrcType>::type;
    using DstEigenCastType = typename EigenCastType<DstType>::type;

    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto in_vector =
        ConstEigenVectorMap<SrcEigenCastType>(reinterpret_cast<const SrcEigenCastType*>(in.Data<SrcType>()), shape_size);
    auto out_vector =
        EigenVectorMap<DstEigenCastType>(reinterpret_cast<DstEigenCastType*>(out.MutableData<DstType>()), shape_size);
    out_vector = in_vector.template cast<DstEigenCastType>();
  }
};

// tensor X -> string
template <typename SrcType>
struct TensorCaster<SrcType, std::string> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<std::string>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      CastToString(in_data[i], out_data[i]);
    }
  }
};

// tensor string -> X
template <typename DstType>
struct TensorCaster<std::string, DstType> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<std::string>();
    auto* out_data = out.MutableData<DstType>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      CastFromString(in_data[i], out_data[i]);
    }
  }
};

// tensor MLFloat16 -> float
template <>
struct TensorCaster<MLFloat16, float> {
  void Cast(const OpKernelContext& ctx, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    auto out_data = out.MutableData<float>();
    auto in_data = in.Data<MLFloat16>();
    const size_t shape_size = narrow<size_t>(shape.Size());
    MlasConvertHalfToFloatBufferInParallel(in_data, out_data, shape_size, ctx.GetOperatorThreadPool());
  }
};

template <typename DstType>
struct TensorCaster<Int4x2, DstType,
                    typename std::enable_if<IsStandardIntegerType<DstType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<DstType>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size, "The Int4x2 tensor size is invalid for casting.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);
      int8_t high_nibble = static_cast<int8_t>(packed) >> 4;
      int8_t low_nibble = static_cast<int8_t>(packed << 4) >> 4;
      out_data[2 * i] = Int4ElementConverter<DstType>::Convert(low_nibble);
      out_data[2 * i + 1] = Int4ElementConverter<DstType>::Convert(high_nibble);
    }
  }
};

template <typename DstType>
struct TensorCaster<Int4x2, DstType,
                    typename std::enable_if<IsStandardFloatType<DstType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<DstType>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size, "The Int4x2 tensor size is invalid for casting.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);

      // Extract signed high and low nibble
      int8_t high_nibble = static_cast<int8_t>(packed) >> 4;
      int8_t low_nibble = static_cast<int8_t>(packed << 4) >> 4;

      // Low nibble first, then high nibble
      out_data[2 * i] = static_cast<DstType>(low_nibble);
      out_data[2 * i + 1] = static_cast<DstType>(high_nibble);
    }
  }
};

template <typename DstType>
struct TensorCaster<Int4x2, DstType,
                    typename std::enable_if<IsOrtFloat16Type<DstType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<DstType>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size, "The Int4x2 tensor size is invalid for casting.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);
      int8_t high_nibble = static_cast<int8_t>(packed) >> 4;
      int8_t low_nibble = static_cast<int8_t>(packed << 4) >> 4;
      out_data[2 * i] = Int4ElementConverter<DstType>::Convert(low_nibble);
      out_data[2 * i + 1] = Int4ElementConverter<DstType>::Convert(high_nibble);
    }
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)
template <typename DstType>
struct TensorCaster<Int4x2, DstType,
                    typename std::enable_if<IsOrtFloat8Type<DstType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<DstType>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size, "The Int4x2 tensor size is invalid for casting.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);
      int8_t high_nibble = static_cast<int8_t>(packed) >> 4;
      int8_t low_nibble = static_cast<int8_t>(packed << 4) >> 4;
      out_data[2 * i] = Int4ElementConverter<DstType>::Convert(low_nibble);
      out_data[2 * i + 1] = Int4ElementConverter<DstType>::Convert(high_nibble);
    }
  }
};
#endif

template <>
struct TensorCaster<Int4x2, bool, void> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<bool>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size, "The Int4x2 tensor size is invalid for casting.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);
      int8_t high_nibble = static_cast<int8_t>(packed) >> 4;
      int8_t low_nibble = static_cast<int8_t>(packed << 4) >> 4;
      out_data[2 * i] = low_nibble != 0;
      out_data[2 * i + 1] = high_nibble != 0;
    }
  }
};

template <>
struct TensorCaster<Int4x2, std::string, void> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<std::string>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size, "The Int4x2 tensor size is invalid for casting.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);
      int8_t high_nibble = static_cast<int8_t>(packed) >> 4;
      int8_t low_nibble = static_cast<int8_t>(packed << 4) >> 4;
      out_data[2 * i] = Int4ElementConverter<std::string>::Convert(low_nibble);
      out_data[2 * i + 1] = Int4ElementConverter<std::string>::Convert(high_nibble);
    }
  }
};

template <typename DstType>
struct TensorCaster<UInt4x2, DstType,
                    typename std::enable_if<IsStandardIntegerType<DstType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<DstType>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size, "The UInt4x2 tensor size is invalid for casting.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);
      uint8_t high_nibble = (packed >> 4) & 0x0F;
      uint8_t low_nibble = packed & 0x0F;
      out_data[2 * i] = Int4ElementConverter<DstType>::Convert(low_nibble);
      out_data[2 * i + 1] = Int4ElementConverter<DstType>::Convert(high_nibble);
    }
  }
};

template <typename DstType>
struct TensorCaster<UInt4x2, DstType,
                    typename std::enable_if<IsStandardFloatType<DstType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<DstType>();

    // Confirm we can unpack the uint4
    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size,
                "The UInt4x2 tensor size is invalid for casting to float.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);

      // Extract unsigned high and low nibble
      uint8_t high_nibble = (packed >> 4) & 0x0F;
      uint8_t low_nibble = packed & 0x0F;

      // Low nibble first, then high nibble
      out_data[2 * i] = static_cast<DstType>(low_nibble);
      out_data[2 * i + 1] = static_cast<DstType>(high_nibble);
    }
  }
};

template <typename DstType>
struct TensorCaster<UInt4x2, DstType,
                    typename std::enable_if<IsOrtFloat16Type<DstType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<DstType>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size, "The UInt4x2 tensor size is invalid for casting.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);
      uint8_t high_nibble = (packed >> 4) & 0x0F;
      uint8_t low_nibble = packed & 0x0F;
      out_data[2 * i] = Int4ElementConverter<DstType>::Convert(low_nibble);
      out_data[2 * i + 1] = Int4ElementConverter<DstType>::Convert(high_nibble);
    }
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)
template <typename DstType>
struct TensorCaster<UInt4x2, DstType,
                    typename std::enable_if<IsOrtFloat8Type<DstType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<DstType>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size, "The UInt4x2 tensor size is invalid for casting.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);
      uint8_t high_nibble = (packed >> 4) & 0x0F;
      uint8_t low_nibble = packed & 0x0F;
      out_data[2 * i] = Int4ElementConverter<DstType>::Convert(low_nibble);
      out_data[2 * i + 1] = Int4ElementConverter<DstType>::Convert(high_nibble);
    }
  }
};
#endif

template <>
struct TensorCaster<UInt4x2, bool, void> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<bool>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size, "The UInt4x2 tensor size is invalid for casting.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);
      uint8_t high_nibble = (packed >> 4) & 0x0F;
      uint8_t low_nibble = packed & 0x0F;
      out_data[2 * i] = low_nibble != 0;
      out_data[2 * i + 1] = high_nibble != 0;
    }
  }
};

template <>
struct TensorCaster<UInt4x2, std::string, void> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<std::string>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size, "The UInt4x2 tensor size is invalid for casting.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);
      uint8_t high_nibble = (packed >> 4) & 0x0F;
      uint8_t low_nibble = packed & 0x0F;
      out_data[2 * i] = Int4ElementConverter<std::string>::Convert(low_nibble);
      out_data[2 * i + 1] = Int4ElementConverter<std::string>::Convert(high_nibble);
    }
  }
};

template <typename SrcType>
struct TensorCaster<SrcType, Int4x2,
                    typename std::enable_if<IsStandardIntegerType<SrcType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<Int4x2>();

    const size_t in_shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(out_shape_size * 2 >= in_shape_size,
                "The output Int4x2 tensor size is invalid for casting from ", typeid(SrcType).name());

    size_t i = 0;
    for (; i < in_shape_size - 1; i += 2) {
      int8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToInt4(in_data[i]);
      int8_t high_val = ToInt4ElementConverter<SrcType>::ConvertToInt4(in_data[i + 1]);
      out_data[i / 2] = Int4x2(low_val, high_val);
    }

    if (i < in_shape_size) {
      int8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToInt4(in_data[i]);
      out_data[i / 2] = Int4x2(low_val, 0);
    }
  }
};

template <typename SrcType>
struct TensorCaster<SrcType, Int4x2,
                    typename std::enable_if<IsStandardFloatType<SrcType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<Int4x2>();

    const size_t in_shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(out_shape_size * 2 >= in_shape_size,
                "The output Int4x2 tensor size is invalid for casting from ", typeid(SrcType).name());

    size_t i = 0;
    for (; i < in_shape_size - 1; i += 2) {
      int8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToInt4(in_data[i]);
      int8_t high_val = ToInt4ElementConverter<SrcType>::ConvertToInt4(in_data[i + 1]);
      out_data[i / 2] = Int4x2(low_val, high_val);
    }

    if (i < in_shape_size) {
      int8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToInt4(in_data[i]);
      out_data[i / 2] = Int4x2(low_val, 0);
    }
  }
};

template <typename SrcType>
struct TensorCaster<SrcType, Int4x2,
                    typename std::enable_if<IsOrtFloat16Type<SrcType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<Int4x2>();

    const size_t in_shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(out_shape_size * 2 >= in_shape_size,
                "The output Int4x2 tensor size is invalid for casting from ", typeid(SrcType).name());

    size_t i = 0;
    for (; i < in_shape_size - 1; i += 2) {
      int8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToInt4(in_data[i]);
      int8_t high_val = ToInt4ElementConverter<SrcType>::ConvertToInt4(in_data[i + 1]);
      out_data[i / 2] = Int4x2(low_val, high_val);
    }

    if (i < in_shape_size) {
      int8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToInt4(in_data[i]);
      out_data[i / 2] = Int4x2(low_val, 0);
    }
  }
};

template <>
struct TensorCaster<bool, Int4x2, void> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<bool>();
    auto* out_data = out.MutableData<Int4x2>();

    const size_t in_shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(out_shape_size * 2 >= in_shape_size,
                "The output Int4x2 tensor size is invalid for casting from bool.");

    size_t i = 0;
    for (; i < in_shape_size - 1; i += 2) {
      int8_t low_val = in_data[i] ? 1 : 0;
      int8_t high_val = in_data[i + 1] ? 1 : 0;
      out_data[i / 2] = Int4x2(low_val, high_val);
    }

    if (i < in_shape_size) {
      int8_t low_val = in_data[i] ? 1 : 0;
      out_data[i / 2] = Int4x2(low_val, 0);
    }
  }
};

template <>
struct TensorCaster<std::string, Int4x2, void> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<std::string>();
    auto* out_data = out.MutableData<Int4x2>();

    const size_t in_shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(out_shape_size * 2 >= in_shape_size,
                "The output Int4x2 tensor size is invalid for casting from string.");

    size_t i = 0;
    for (; i < in_shape_size - 1; i += 2) {
      int8_t low_val = ToInt4ElementConverter<std::string>::ConvertToInt4(in_data[i]);
      int8_t high_val = ToInt4ElementConverter<std::string>::ConvertToInt4(in_data[i + 1]);
      out_data[i / 2] = Int4x2(low_val, high_val);
    }

    if (i < in_shape_size) {
      int8_t low_val = ToInt4ElementConverter<std::string>::ConvertToInt4(in_data[i]);
      out_data[i / 2] = Int4x2(low_val, 0);
    }
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)
template <typename SrcType>
struct TensorCaster<SrcType, Int4x2,
                    typename std::enable_if<IsOrtFloat8Type<SrcType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<Int4x2>();

    const size_t in_shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(out_shape_size * 2 >= in_shape_size,
                "The output Int4x2 tensor size is invalid for casting from ", typeid(SrcType).name());

    size_t i = 0;
    for (; i < in_shape_size - 1; i += 2) {
      int8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToInt4(in_data[i]);
      int8_t high_val = ToInt4ElementConverter<SrcType>::ConvertToInt4(in_data[i + 1]);
      out_data[i / 2] = Int4x2(low_val, high_val);
    }

    if (i < in_shape_size) {
      int8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToInt4(in_data[i]);
      out_data[i / 2] = Int4x2(low_val, 0);
    }
  }
};
#endif


template <typename SrcType>
struct TensorCaster<SrcType, UInt4x2,
                    typename std::enable_if<IsStandardIntegerType<SrcType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<UInt4x2>();

    const size_t in_shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(out_shape_size * 2 >= in_shape_size,
                "The output UInt4x2 tensor size is invalid for casting from ", typeid(SrcType).name());

    size_t i = 0;
    for (; i < in_shape_size - 1; i += 2) {
      uint8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToUInt4(in_data[i]);
      uint8_t high_val = ToInt4ElementConverter<SrcType>::ConvertToUInt4(in_data[i + 1]);
      out_data[i / 2] = UInt4x2(low_val, high_val);
    }

    if (i < in_shape_size) {
      uint8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToUInt4(in_data[i]);
      out_data[i / 2] = UInt4x2(low_val, 0);
    }
  }
};

template <typename SrcType>
struct TensorCaster<SrcType, UInt4x2,
                    typename std::enable_if<IsStandardFloatType<SrcType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<UInt4x2>();

    const size_t in_shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(out_shape_size * 2 >= in_shape_size,
                "The output UInt4x2 tensor size is invalid for casting from ", typeid(SrcType).name());

    size_t i = 0;
    for (; i < in_shape_size - 1; i += 2) {
      uint8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToUInt4(in_data[i]);
      uint8_t high_val = ToInt4ElementConverter<SrcType>::ConvertToUInt4(in_data[i + 1]);
      out_data[i / 2] = UInt4x2(low_val, high_val);
    }

    if (i < in_shape_size) {
      uint8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToUInt4(in_data[i]);
      out_data[i / 2] = UInt4x2(low_val, 0);
    }
  }
};

template <typename SrcType>
struct TensorCaster<SrcType, UInt4x2,
                    typename std::enable_if<IsOrtFloat16Type<SrcType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<UInt4x2>();

    const size_t in_shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(out_shape_size * 2 >= in_shape_size,
                "The output UInt4x2 tensor size is invalid for casting from ", typeid(SrcType).name());

    size_t i = 0;
    for (; i < in_shape_size - 1; i += 2) {
      uint8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToUInt4(in_data[i]);
      uint8_t high_val = ToInt4ElementConverter<SrcType>::ConvertToUInt4(in_data[i + 1]);
      out_data[i / 2] = UInt4x2(low_val, high_val);
    }

    if (i < in_shape_size) {
      uint8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToUInt4(in_data[i]);
      out_data[i / 2] = UInt4x2(low_val, 0);
    }
  }
};

template <>
struct TensorCaster<bool, UInt4x2, void> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<bool>();
    auto* out_data = out.MutableData<UInt4x2>();

    const size_t in_shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(out_shape_size * 2 >= in_shape_size,
                "The output UInt4x2 tensor size is invalid for casting from bool.");

    size_t i = 0;
    for (; i < in_shape_size - 1; i += 2) {
      uint8_t low_val = in_data[i] ? 1 : 0;
      uint8_t high_val = in_data[i + 1] ? 1 : 0;
      out_data[i / 2] = UInt4x2(low_val, high_val);
    }

    if (i < in_shape_size) {
      uint8_t low_val = in_data[i] ? 1 : 0;
      out_data[i / 2] = UInt4x2(low_val, 0);
    }
  }
};

template <>
struct TensorCaster<std::string, UInt4x2, void> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<std::string>();
    auto* out_data = out.MutableData<UInt4x2>();

    const size_t in_shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(out_shape_size * 2 >= in_shape_size,
                "The output UInt4x2 tensor size is invalid for casting from string.");

    size_t i = 0;
    for (; i < in_shape_size - 1; i += 2) {
      uint8_t low_val = ToInt4ElementConverter<std::string>::ConvertToUInt4(in_data[i]);
      uint8_t high_val = ToInt4ElementConverter<std::string>::ConvertToUInt4(in_data[i + 1]);
      out_data[i / 2] = UInt4x2(low_val, high_val);
    }

    if (i < in_shape_size) {
      uint8_t low_val = ToInt4ElementConverter<std::string>::ConvertToUInt4(in_data[i]);
      out_data[i / 2] = UInt4x2(low_val, 0);
    }
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)
template <typename SrcType>
struct TensorCaster<SrcType, UInt4x2,
                    typename std::enable_if<IsOrtFloat8Type<SrcType>::value>::type> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<UInt4x2>();

    const size_t in_shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(out_shape_size * 2 >= in_shape_size,
                "The output UInt4x2 tensor size is invalid for casting from ", typeid(SrcType).name());

    size_t i = 0;
    for (; i < in_shape_size - 1; i += 2) {
      uint8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToUInt4(in_data[i]);
      uint8_t high_val = ToInt4ElementConverter<SrcType>::ConvertToUInt4(in_data[i + 1]);
      out_data[i / 2] = UInt4x2(low_val, high_val);
    }

    if (i < in_shape_size) {
      uint8_t low_val = ToInt4ElementConverter<SrcType>::ConvertToUInt4(in_data[i]);
      out_data[i / 2] = UInt4x2(low_val, 0);
    }
  }
};
#endif

#if defined(_M_AMD64) && !defined(_M_ARM64EC)
// specializations to use optimized and Windows x64-specific

Tensor GetIntermediateMLFloat16ToFloatTensor(
    const OpKernelContext& context, const TensorShape& shape, const Tensor& in) {
  AllocatorPtr allocator;
  ORT_THROW_IF_ERROR(context.GetTempSpaceAllocator(&allocator));
  Tensor out{DataTypeImpl::GetType<float>(), shape, allocator};
  TensorCaster<MLFloat16, float>{}.Cast(context, shape, in, out);
  return out;
}

template <typename DstType>
void CastMLFloat16ThroughFloatTensor(
    const OpKernelContext& context, const TensorShape& shape, const Tensor& in, Tensor& out) {
  // use optimized MLFloat16 -> float, then float -> DstType
  Tensor intermediate_tensor = GetIntermediateMLFloat16ToFloatTensor(context, shape, in);
  TensorCaster<float, DstType>{}.Cast(context, shape, intermediate_tensor, out);
}

// tensor MLFloat16 -> X
template <typename DstType>
struct TensorCaster<MLFloat16, DstType> {
  void Cast(const OpKernelContext& context, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    CastMLFloat16ThroughFloatTensor<DstType>(context, shape, in, out);
  }
};

// tensor MLFloat16 -> string
template <>
struct TensorCaster<MLFloat16, std::string> {
  void Cast(const OpKernelContext& context, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    CastMLFloat16ThroughFloatTensor<std::string>(context, shape, in, out);
  }
};
#endif

#if !defined(DISABLE_FLOAT8_TYPES)

// tensor X -> float 8
template <typename SrcType, typename DstType, typename Enable = void>
struct TensorCasterNoSat {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<DstType>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      out_data[i] = DstType(static_cast<float>(in_data[i]), false);
    }
  }
};

// TensorCasterNoSat should never be instantiated for Int4x2/UInt4x2
template <typename DstType>
struct TensorCasterNoSat<Int4x2, DstType, void> {
  void Cast(const OpKernelContext&, const TensorShape&, const Tensor&, Tensor&) const {
    ORT_THROW("Int4x2 should never use TensorCasterNoSat");
  }
};

template <typename DstType>
struct TensorCasterNoSat<UInt4x2, DstType, void> {
  void Cast(const OpKernelContext&, const TensorShape&, const Tensor&, Tensor&) const {
    ORT_THROW("UInt4x2 should never use TensorCasterNoSat");
  }
};

// tensor string -> float 8
template <typename DstType>
struct TensorCasterNoSat<std::string, DstType> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<std::string>();
    auto* out_data = out.MutableData<DstType>();
    float float_value;
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      CastFromString(in_data[i], float_value);
      out_data[i] = DstType(float_value, false);
    }
  }
};

#endif

class Cast final : public OpKernel {
 public:
  Cast(const OpKernelInfo& info) : OpKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);

    int64_t saturate = info.GetAttrOrDefault("saturate", int64_t{1});
#if !defined(DISABLE_FLOAT8_TYPES)
    if (saturate == 0 && (to != ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN &&
                          to != ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FNUZ &&
                          to != ONNX_NAMESPACE::TensorProto::FLOAT8E5M2 &&
                          to != ONNX_NAMESPACE::TensorProto::FLOAT8E5M2FNUZ)) {
      ORT_THROW("Attribute saturate is only used for cast to float 8 types.");
    }
#else
    if (saturate == 0) {
      ORT_THROW("Attribute saturate is only used for cast to float 8 types.");
    }
#endif
    saturate_ = saturate == 1;
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType to_;
  bool saturate_;
};

template <typename TSrc, typename TDst>
struct Dispatcher {
  void operator()(const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) {
    TensorCaster<TSrc, TDst>{}.Cast(context, shape, src, dst);
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename TSrc, typename TDst>
struct DispatcherNoSat {
  void operator()(const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) {
    TensorCasterNoSat<TSrc, TDst>{}.Cast(context, shape, src, dst);
  }
};

#endif

template <typename TSrc>
struct SrcDispatcher {
  void operator()(
      int32_t to, const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) {
    using EnabledDstTypesWithoutSrcType =
        boost::mp11::mp_remove_if_q<EnabledDstTypes, boost::mp11::mp_bind_front<std::is_same, TSrc>>;
    utils::MLTypeCallDispatcherFromTypeList<EnabledDstTypesWithoutSrcType> dispatcher{to};
    dispatcher.template InvokeWithLeadingTemplateArgs<Dispatcher, TypeList<TSrc>>(context, shape, src, dst);
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename TSrc>
struct SrcDispatcherNoSat {
  void operator()(
      int32_t to, const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) {
    using EnabledDstTypeOnlyFloat8 = boost::mp11::mp_set_intersection<
        EnabledDstTypes, element_type_lists::AllFloat8>;
    using EnabledDstTypesWithoutSrcType =
        boost::mp11::mp_remove_if_q<EnabledDstTypeOnlyFloat8, boost::mp11::mp_bind_front<std::is_same, TSrc>>;
    utils::MLTypeCallDispatcherFromTypeList<EnabledDstTypesWithoutSrcType> dispatcher{to};
    dispatcher.template InvokeWithLeadingTemplateArgs<DispatcherNoSat, TypeList<TSrc>>(context, shape, src, dst);
  }
};

#endif

Status Cast::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);

  if (shape.Size() == 0) {
    return Status::OK();
  }

  const auto from = X->GetElementType();

  if (from == to_) {
    // will copy if X and Y have different buffers
    CopyCpuTensor(X, Y);
    return Status::OK();
  }

#if !defined(DISABLE_FLOAT8_TYPES)
  if (saturate_) {
#endif
    utils::MLTypeCallDispatcherFromTypeList<EnabledSrcTypes> dispatcher{from};
    dispatcher.Invoke<SrcDispatcher>(to_, *context, shape, *X, *Y);
#if !defined(DISABLE_FLOAT8_TYPES)
  } else if (to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN ||
             to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FNUZ ||
             to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E5M2 ||
             to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E5M2FNUZ) {
    utils::MLTypeCallDispatcherFromTypeList<EnabledSrcTypes> dispatcher{from};
    dispatcher.Invoke<SrcDispatcherNoSat>(to_, *context, shape, *X, *Y);
  }
#endif

  return Status::OK();
}
}  // namespace

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    6,
    12,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypes>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypes>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    13,
    18,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypes>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypes>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    19,
    20,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypes>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypes>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

// TODO(adrianlizarraga): Implement support for int4 and uint4.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    21,
    22,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypes>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypes>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

// Opset 23 added support for float4e2m1.
// TODO(titaiwang): Implement support for float4e2m1.
ONNX_CPU_OPERATOR_KERNEL(
    Cast,
    23,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypes>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypes>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

}  // namespace onnxruntime
