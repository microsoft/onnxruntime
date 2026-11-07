// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "core/providers/cpu/nn/shrink.h"

#include "core/framework/element_type_lists.h"
#include "core/framework/math.h"
#include "core/framework/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Shrink, Input, 0,
    element_type_lists::AllNumeric);
}

using EnabledShrinkDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Shrink, Input, 0);

ONNX_CPU_OPERATOR_KERNEL(
    Shrink,
    9,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledShrinkDataTypes>()),
    Shrink);
// TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26451)
#endif

namespace shrink_internal {

template <typename T>
T ClampIntegerMagnitude(uint64_t magnitude, bool negative) {
  if constexpr (std::is_signed_v<T>) {
    if (negative) {
      const uint64_t limit = uint64_t{0} - static_cast<uint64_t>(std::numeric_limits<T>::lowest());
      if (magnitude >= limit) {
        return std::numeric_limits<T>::lowest();
      }
      return static_cast<T>(-static_cast<int64_t>(magnitude));
    }
  } else if (negative) {
    return 0;
  }

  if (magnitude > static_cast<uint64_t>(std::numeric_limits<T>::max())) {
    return std::numeric_limits<T>::max();
  }
  return static_cast<T>(magnitude);
}

template <typename T>
T ApplyIntegerBias(T value, float bias, bool add_bias) {
  const double offset = add_bias ? static_cast<double>(bias) : -static_cast<double>(bias);
  if (std::isnan(offset)) {
    return 0;
  }

  const bool offset_negative = std::signbit(offset);
  const double offset_magnitude = std::fabs(offset);
  constexpr double kUint64MaxExclusive = 18446744073709551616.0;  // 2^64
  if (offset_magnitude >= kUint64MaxExclusive) {
    return ClampIntegerMagnitude<T>(std::numeric_limits<uint64_t>::max(), offset_negative);
  }

  double integral_part;
  const bool has_fraction = std::modf(offset_magnitude, &integral_part) != 0.0;
  const uint64_t offset_integer = static_cast<uint64_t>(integral_part);
  uint64_t value_magnitude = static_cast<uint64_t>(value);
  bool value_negative = false;
  if constexpr (std::is_signed_v<T>) {
    if (value < 0) {
      value_negative = true;
      value_magnitude = uint64_t{0} - value_magnitude;
    }
  }

  // Keep the integer and fractional parts separate so values above 2^53 stay exact.
  if (value_negative == offset_negative) {
    if (offset_integer > std::numeric_limits<uint64_t>::max() - value_magnitude) {
      return ClampIntegerMagnitude<T>(std::numeric_limits<uint64_t>::max(), value_negative);
    }
    return ClampIntegerMagnitude<T>(value_magnitude + offset_integer, value_negative);
  }

  if (value_magnitude > offset_integer) {
    return ClampIntegerMagnitude<T>(value_magnitude - offset_integer - static_cast<uint64_t>(has_fraction), value_negative);
  }
  if (value_magnitude < offset_integer) {
    return ClampIntegerMagnitude<T>(offset_integer - value_magnitude, offset_negative);
  }
  return 0;
}

template <class T>
inline T ShrinkCore(const T& t_val, float bias, float lambd) {
  if constexpr (std::is_same_v<T, float>) {
    if (t_val < -lambd) {
      return t_val + bias;
    }
    if (t_val > lambd) {
      return t_val - bias;
    }
    return 0.f;
  } else if constexpr (std::is_same_v<T, double>) {
    const double d_lambd = static_cast<double>(lambd);
    const double d_bias = static_cast<double>(bias);
    if (t_val < -d_lambd) {
      return t_val + d_bias;
    }
    if (t_val > d_lambd) {
      return t_val - d_bias;
    }
    return 0.0;
  } else if constexpr (std::is_same_v<T, MLFloat16> || std::is_same_v<T, BFloat16>) {
    float val = static_cast<float>(t_val);
    if (val < -lambd) {
      return static_cast<T>(val + bias);
    }
    if (val > lambd) {
      return static_cast<T>(val - bias);
    }
    return static_cast<T>(0.f);
  } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    if (std::isnan(lambd)) {
      return static_cast<T>(0);
    }
    constexpr double kMaxSigned = 9223372036854775808.0;  // 2^63
    const double k = -static_cast<double>(lambd);
    bool c1 = false;
    const double d_ceil = std::ceil(k);
    if (d_ceil >= kMaxSigned) {
      c1 = true;
    } else if (d_ceil <= -kMaxSigned) {
      c1 = false;
    } else {
      c1 = (static_cast<int64_t>(t_val) < static_cast<int64_t>(d_ceil));
    }

    if (c1) {
      return ApplyIntegerBias(t_val, bias, true);
    }

    const double d_lambd = static_cast<double>(lambd);
    bool c2 = false;
    const double d_floor = std::floor(d_lambd);
    if (d_floor >= kMaxSigned) {
      c2 = false;
    } else if (d_floor < -kMaxSigned) {
      c2 = true;
    } else {
      c2 = (static_cast<int64_t>(t_val) > static_cast<int64_t>(d_floor));
    }

    if (c2) {
      return ApplyIntegerBias(t_val, bias, false);
    }

    return static_cast<T>(0);
  } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
    if (std::isnan(lambd)) {
      return static_cast<T>(0);
    }
    constexpr double kMaxUnsigned = 18446744073709551616.0;  // 2^64
    const double k = -static_cast<double>(lambd);
    bool c1 = false;
    if (k > 0.0) {
      const double d_ceil = std::ceil(k);
      if (d_ceil >= kMaxUnsigned) {
        c1 = true;
      } else {
        c1 = (static_cast<uint64_t>(t_val) < static_cast<uint64_t>(d_ceil));
      }
    }

    if (c1) {
      return ApplyIntegerBias(t_val, bias, true);
    }

    const double d_lambd = static_cast<double>(lambd);
    bool c2 = false;
    if (d_lambd < 0.0) {
      c2 = true;
    } else {
      const double d_floor = std::floor(d_lambd);
      if (d_floor >= kMaxUnsigned) {
        c2 = false;
      } else {
        c2 = (static_cast<uint64_t>(t_val) > static_cast<uint64_t>(d_floor));
      }
    }

    if (c2) {
      return ApplyIntegerBias(t_val, bias, false);
    }

    return static_cast<T>(0);
  } else {
    return static_cast<T>(0);
  }
}

template <class T>
struct CallShrinkImpl {
  Status operator()(const Tensor* input, Tensor* output, float bias, float lambd) const {
    EigenMap<T>(*output) = EigenMap<T>(*input).unaryExpr([bias, lambd](const T& val) { return ShrinkCore<T>(val, bias, lambd); });
    return Status::OK();
  }
};

}  // namespace shrink_internal

Status Shrink::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* input = p_op_kernel_context->Input<Tensor>(0);
  auto* output = p_op_kernel_context->Output(0, input->Shape());
  utils::MLTypeCallDispatcherFromTypeList<EnabledShrinkDataTypes> t_disp(input->GetElementType());
  return t_disp.InvokeRet<Status, shrink_internal::CallShrinkImpl>(input, output, bias_, lambd_);
}
}  // namespace onnxruntime
