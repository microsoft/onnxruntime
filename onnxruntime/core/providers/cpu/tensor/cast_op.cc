// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>
#include <algorithm>

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
#include "core/framework/int4.h"
#if !defined(DISABLE_FLOAT8_TYPES)
#include "core/framework/float8.h"
#endif

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
// generic tensor X -> Y (Int4x2 and UInt4x2 have specialized implementations)
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

// tensor Int4x2 -> float
template <>
struct TensorCaster<Int4x2, float> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<float>();

    // Confirm we can unpack the int4
    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size,
                "The Int4x2 tensor size is invalid for casting to float.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);

      // Extract signed high and low nibble
      int8_t high_nibble = static_cast<int8_t>(packed) >> 4;
      int8_t low_nibble = static_cast<int8_t>(packed << 4) >> 4;

      out_data[2 * i] = static_cast<float>(low_nibble);
      out_data[2 * i + 1] = static_cast<float>(high_nibble);
    }
  }
};

// tensor UInt4x2 -> float
template <>
struct TensorCaster<UInt4x2, float> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<float>();

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

      out_data[2 * i] = static_cast<float>(low_nibble);
      out_data[2 * i + 1] = static_cast<float>(high_nibble);
    }
  }
};

// Helper macro to define Int4x2 -> DstType casting
#define DEFINE_INT4X2_CAST(DstType) \
template <> \
struct TensorCaster<Int4x2, DstType> { \
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const { \
    const auto* in_data = in.Data<Int4x2>(); \
    auto* out_data = out.MutableData<DstType>(); \
    \
    const size_t shape_size = narrow<size_t>(shape.Size()); \
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size()); \
    ORT_ENFORCE(in_shape_size * 2 == shape_size, \
                "The Int4x2 tensor size is invalid for casting to " #DstType "."); \
    \
    for (size_t i = 0; i < in_shape_size; ++i) { \
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_); \
      \
      /* Extract signed high and low nibble */ \
      int8_t high_nibble = static_cast<int8_t>(packed) >> 4; \
      int8_t low_nibble = static_cast<int8_t>(packed << 4) >> 4; \
      \
      if constexpr (std::is_same_v<DstType, MLFloat16> || std::is_same_v<DstType, BFloat16>) { \
        out_data[2 * i] = static_cast<DstType>(static_cast<float>(low_nibble)); \
        out_data[2 * i + 1] = static_cast<DstType>(static_cast<float>(high_nibble)); \
      } else { \
        out_data[2 * i] = static_cast<DstType>(low_nibble); \
        out_data[2 * i + 1] = static_cast<DstType>(high_nibble); \
      } \
    } \
  } \
};

// Helper macro to define UInt4x2 -> DstType casting
#define DEFINE_UINT4X2_CAST(DstType) \
template <> \
struct TensorCaster<UInt4x2, DstType> { \
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const { \
    const auto* in_data = in.Data<UInt4x2>(); \
    auto* out_data = out.MutableData<DstType>(); \
    \
    const size_t shape_size = narrow<size_t>(shape.Size()); \
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size()); \
    ORT_ENFORCE(in_shape_size * 2 == shape_size, \
                "The UInt4x2 tensor size is invalid for casting to " #DstType "."); \
    \
    for (size_t i = 0; i < in_shape_size; ++i) { \
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_); \
      \
      /* Extract unsigned high and low nibble */ \
      uint8_t high_nibble = (packed >> 4) & 0x0F; \
      uint8_t low_nibble = packed & 0x0F; \
      \
      if constexpr (std::is_same_v<DstType, MLFloat16> || std::is_same_v<DstType, BFloat16>) { \
        out_data[2 * i] = static_cast<DstType>(static_cast<float>(low_nibble)); \
        out_data[2 * i + 1] = static_cast<DstType>(static_cast<float>(high_nibble)); \
      } else { \
        out_data[2 * i] = static_cast<DstType>(low_nibble); \
        out_data[2 * i + 1] = static_cast<DstType>(high_nibble); \
      } \
    } \
  } \
};

// Define casts for all common destination types
DEFINE_INT4X2_CAST(double)
DEFINE_INT4X2_CAST(int64_t)
DEFINE_INT4X2_CAST(uint64_t)
DEFINE_INT4X2_CAST(int32_t)
DEFINE_INT4X2_CAST(uint32_t)
DEFINE_INT4X2_CAST(int16_t)
DEFINE_INT4X2_CAST(uint16_t)
DEFINE_INT4X2_CAST(int8_t)
DEFINE_INT4X2_CAST(uint8_t)
DEFINE_INT4X2_CAST(bool)

// Specialized casts for MLFloat16 and BFloat16 (need explicit float conversion)
template <>
struct TensorCaster<Int4x2, MLFloat16> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<MLFloat16>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size,
                "The Int4x2 tensor size is invalid for casting to MLFloat16.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);

      // Extract signed high and low nibble
      int8_t high_nibble = static_cast<int8_t>(packed) >> 4;
      int8_t low_nibble = static_cast<int8_t>(packed << 4) >> 4;

      out_data[2 * i] = MLFloat16(static_cast<float>(low_nibble));
      out_data[2 * i + 1] = MLFloat16(static_cast<float>(high_nibble));
    }
  }
};

template <>
struct TensorCaster<Int4x2, BFloat16> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<BFloat16>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size,
                "The Int4x2 tensor size is invalid for casting to BFloat16.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);

      // Extract signed high and low nibble
      int8_t high_nibble = static_cast<int8_t>(packed) >> 4;
      int8_t low_nibble = static_cast<int8_t>(packed << 4) >> 4;

      out_data[2 * i] = BFloat16(static_cast<float>(low_nibble));
      out_data[2 * i + 1] = BFloat16(static_cast<float>(high_nibble));
    }
  }
};

DEFINE_UINT4X2_CAST(double)
DEFINE_UINT4X2_CAST(int64_t)
DEFINE_UINT4X2_CAST(uint64_t)
DEFINE_UINT4X2_CAST(int32_t)
DEFINE_UINT4X2_CAST(uint32_t)
DEFINE_UINT4X2_CAST(int16_t)
DEFINE_UINT4X2_CAST(uint16_t)
DEFINE_UINT4X2_CAST(int8_t)
DEFINE_UINT4X2_CAST(uint8_t)
DEFINE_UINT4X2_CAST(bool)

// Specialized casts for MLFloat16 and BFloat16 (need explicit float conversion)
template <>
struct TensorCaster<UInt4x2, MLFloat16> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<MLFloat16>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size,
                "The UInt4x2 tensor size is invalid for casting to MLFloat16.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);

      // Extract unsigned high and low nibble
      uint8_t high_nibble = (packed >> 4) & 0x0F;
      uint8_t low_nibble = packed & 0x0F;

      out_data[2 * i] = MLFloat16(static_cast<float>(low_nibble));
      out_data[2 * i + 1] = MLFloat16(static_cast<float>(high_nibble));
    }
  }
};

template <>
struct TensorCaster<UInt4x2, BFloat16> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<BFloat16>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t in_shape_size = narrow<size_t>(in.Shape().Size());
    ORT_ENFORCE(in_shape_size * 2 == shape_size,
                "The UInt4x2 tensor size is invalid for casting to BFloat16.");

    for (size_t i = 0; i < in_shape_size; ++i) {
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);

      // Extract unsigned high and low nibble
      uint8_t high_nibble = (packed >> 4) & 0x0F;
      uint8_t low_nibble = packed & 0x0F;

      out_data[2 * i] = BFloat16(static_cast<float>(low_nibble));
      out_data[2 * i + 1] = BFloat16(static_cast<float>(high_nibble));
    }
  }
};

// Specialized casts for Int4x2 <-> UInt4x2
template <>
struct TensorCaster<Int4x2, UInt4x2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<UInt4x2>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    ORT_ENFORCE(shape_size == narrow<size_t>(in.Shape().Size()) && shape_size == narrow<size_t>(out.Shape().Size()),
                "Int4x2 to UInt4x2 cast requires same tensor sizes.");

    for (size_t i = 0; i < shape_size; ++i) {
      // Convert each signed 4-bit pair to unsigned by treating values as unsigned
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);
      out_data[i] = UInt4x2(std::byte{packed});
    }
  }
};

template <>
struct TensorCaster<UInt4x2, Int4x2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<Int4x2>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    ORT_ENFORCE(shape_size == narrow<size_t>(in.Shape().Size()) && shape_size == narrow<size_t>(out.Shape().Size()),
                "UInt4x2 to Int4x2 cast requires same tensor sizes.");

    for (size_t i = 0; i < shape_size; ++i) {
      // Convert each unsigned 4-bit pair to signed by treating values as signed
      const uint8_t packed = static_cast<uint8_t>(in_data[i].bits_);
      out_data[i] = Int4x2(std::byte{packed});
    }
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)
DEFINE_INT4X2_CAST(Float8E4M3FN)
DEFINE_INT4X2_CAST(Float8E4M3FNUZ)
DEFINE_INT4X2_CAST(Float8E5M2)
DEFINE_INT4X2_CAST(Float8E5M2FNUZ)

DEFINE_UINT4X2_CAST(Float8E4M3FN)
DEFINE_UINT4X2_CAST(Float8E4M3FNUZ)
DEFINE_UINT4X2_CAST(Float8E5M2)
DEFINE_UINT4X2_CAST(Float8E5M2FNUZ)
#endif

#undef DEFINE_INT4X2_CAST
#undef DEFINE_UINT4X2_CAST

// Helper macro to define SrcType -> Int4x2 casting
#define DEFINE_CAST_TO_INT4X2(SrcType) \
template <> \
struct TensorCaster<SrcType, Int4x2> { \
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const { \
    const auto* in_data = in.Data<SrcType>(); \
    auto* out_data = out.MutableData<Int4x2>(); \
    \
    const size_t shape_size = narrow<size_t>(shape.Size()); \
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size()); \
    ORT_ENFORCE(shape_size == out_shape_size * 2, \
                "The input tensor size must be twice the Int4x2 tensor size for casting from " #SrcType "."); \
    \
    for (size_t i = 0; i < out_shape_size; ++i) { \
      /* Pack two consecutive input values into one Int4x2 */ \
      auto val0 = static_cast<int8_t>(std::clamp(static_cast<int>(in_data[2 * i]), -8, 7)); \
      auto val1 = static_cast<int8_t>(std::clamp(static_cast<int>(in_data[2 * i + 1]), -8, 7)); \
      out_data[i] = Int4x2(val0, val1); \
    } \
  } \
};

// Helper macro to define SrcType -> UInt4x2 casting
#define DEFINE_CAST_TO_UINT4X2(SrcType) \
template <> \
struct TensorCaster<SrcType, UInt4x2> { \
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const { \
    const auto* in_data = in.Data<SrcType>(); \
    auto* out_data = out.MutableData<UInt4x2>(); \
    \
    const size_t shape_size = narrow<size_t>(shape.Size()); \
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size()); \
    ORT_ENFORCE(shape_size == out_shape_size * 2, \
                "The input tensor size must be twice the UInt4x2 tensor size for casting from " #SrcType "."); \
    \
    for (size_t i = 0; i < out_shape_size; ++i) { \
      /* Pack two consecutive input values into one UInt4x2 */ \
      auto val0 = static_cast<uint8_t>(std::clamp(static_cast<int>(in_data[2 * i]), 0, 15)); \
      auto val1 = static_cast<uint8_t>(std::clamp(static_cast<int>(in_data[2 * i + 1]), 0, 15)); \
      out_data[i] = UInt4x2(val0, val1); \
    } \
  } \
};

// Define casts from common source types to Int4x2/UInt4x2
DEFINE_CAST_TO_INT4X2(float)
DEFINE_CAST_TO_INT4X2(double)
DEFINE_CAST_TO_INT4X2(int64_t)
DEFINE_CAST_TO_INT4X2(uint64_t)
DEFINE_CAST_TO_INT4X2(int32_t)
DEFINE_CAST_TO_INT4X2(uint32_t)
DEFINE_CAST_TO_INT4X2(int16_t)
DEFINE_CAST_TO_INT4X2(uint16_t)
DEFINE_CAST_TO_INT4X2(int8_t)
DEFINE_CAST_TO_INT4X2(uint8_t)
DEFINE_CAST_TO_INT4X2(bool)

DEFINE_CAST_TO_UINT4X2(float)
DEFINE_CAST_TO_UINT4X2(double)
DEFINE_CAST_TO_UINT4X2(int64_t)
DEFINE_CAST_TO_UINT4X2(uint64_t)
DEFINE_CAST_TO_UINT4X2(int32_t)
DEFINE_CAST_TO_UINT4X2(uint32_t)
DEFINE_CAST_TO_UINT4X2(int16_t)
DEFINE_CAST_TO_UINT4X2(uint16_t)
DEFINE_CAST_TO_UINT4X2(int8_t)
DEFINE_CAST_TO_UINT4X2(uint8_t)
DEFINE_CAST_TO_UINT4X2(bool)

// Specialized casts from MLFloat16 and BFloat16 to Int4x2/UInt4x2 (need explicit float conversion)
template <>
struct TensorCaster<MLFloat16, Int4x2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<MLFloat16>();
    auto* out_data = out.MutableData<Int4x2>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(shape_size == out_shape_size * 2,
                "The input tensor size must be twice the Int4x2 tensor size for casting from MLFloat16.");

    for (size_t i = 0; i < out_shape_size; ++i) {
      // Pack two consecutive input values into one Int4x2
      auto val0 = static_cast<int8_t>(std::clamp(static_cast<int>(static_cast<float>(in_data[2 * i])), -8, 7));
      auto val1 = static_cast<int8_t>(std::clamp(static_cast<int>(static_cast<float>(in_data[2 * i + 1])), -8, 7));
      out_data[i] = Int4x2(val0, val1);
    }
  }
};

template <>
struct TensorCaster<BFloat16, Int4x2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<BFloat16>();
    auto* out_data = out.MutableData<Int4x2>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(shape_size == out_shape_size * 2,
                "The input tensor size must be twice the Int4x2 tensor size for casting from BFloat16.");

    for (size_t i = 0; i < out_shape_size; ++i) {
      // Pack two consecutive input values into one Int4x2
      auto val0 = static_cast<int8_t>(std::clamp(static_cast<int>(static_cast<float>(in_data[2 * i])), -8, 7));
      auto val1 = static_cast<int8_t>(std::clamp(static_cast<int>(static_cast<float>(in_data[2 * i + 1])), -8, 7));
      out_data[i] = Int4x2(val0, val1);
    }
  }
};

template <>
struct TensorCaster<MLFloat16, UInt4x2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<MLFloat16>();
    auto* out_data = out.MutableData<UInt4x2>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(shape_size == out_shape_size * 2,
                "The input tensor size must be twice the UInt4x2 tensor size for casting from MLFloat16.");

    for (size_t i = 0; i < out_shape_size; ++i) {
      // Pack two consecutive input values into one UInt4x2
      auto val0 = static_cast<uint8_t>(std::clamp(static_cast<int>(static_cast<float>(in_data[2 * i])), 0, 15));
      auto val1 = static_cast<uint8_t>(std::clamp(static_cast<int>(static_cast<float>(in_data[2 * i + 1])), 0, 15));
      out_data[i] = UInt4x2(val0, val1);
    }
  }
};

template <>
struct TensorCaster<BFloat16, UInt4x2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const auto* in_data = in.Data<BFloat16>();
    auto* out_data = out.MutableData<UInt4x2>();

    const size_t shape_size = narrow<size_t>(shape.Size());
    const size_t out_shape_size = narrow<size_t>(out.Shape().Size());
    ORT_ENFORCE(shape_size == out_shape_size * 2,
                "The input tensor size must be twice the UInt4x2 tensor size for casting from BFloat16.");

    for (size_t i = 0; i < out_shape_size; ++i) {
      // Pack two consecutive input values into one UInt4x2
      auto val0 = static_cast<uint8_t>(std::clamp(static_cast<int>(static_cast<float>(in_data[2 * i])), 0, 15));
      auto val1 = static_cast<uint8_t>(std::clamp(static_cast<int>(static_cast<float>(in_data[2 * i + 1])), 0, 15));
      out_data[i] = UInt4x2(val0, val1);
    }
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)
DEFINE_CAST_TO_INT4X2(Float8E4M3FN)
DEFINE_CAST_TO_INT4X2(Float8E4M3FNUZ)
DEFINE_CAST_TO_INT4X2(Float8E5M2)
DEFINE_CAST_TO_INT4X2(Float8E5M2FNUZ)

DEFINE_CAST_TO_UINT4X2(Float8E4M3FN)
DEFINE_CAST_TO_UINT4X2(Float8E4M3FNUZ)
DEFINE_CAST_TO_UINT4X2(Float8E5M2)
DEFINE_CAST_TO_UINT4X2(Float8E5M2FNUZ)
#endif

#undef DEFINE_CAST_TO_INT4X2
#undef DEFINE_CAST_TO_UINT4X2

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
