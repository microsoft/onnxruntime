// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
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

template <typename IntT>
inline IntT SafeFloatToInteger(double val) {
  if (std::isnan(val)) {
    return static_cast<IntT>(0);
  }
  constexpr double kMax = static_cast<double>(std::numeric_limits<IntT>::max());
  constexpr double kMin = static_cast<double>(std::numeric_limits<IntT>::lowest());
  if (val >= kMax) {
    return std::numeric_limits<IntT>::max();
  }
  if (val <= kMin) {
    return std::numeric_limits<IntT>::lowest();
  }
  return static_cast<IntT>(val);
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

    const double d_bias = static_cast<double>(bias);
    const bool is_int_bias = (d_bias >= -kMaxSigned && d_bias < kMaxSigned && d_bias == std::floor(d_bias));
    if (c1) {
      if (is_int_bias) {
        const int64_t b = static_cast<int64_t>(d_bias);
        const uint64_t u_val = static_cast<uint64_t>(static_cast<int64_t>(t_val));
        const uint64_t res = (b >= 0) ? (u_val + static_cast<uint64_t>(b))
                                      : (u_val - static_cast<uint64_t>(-b));
        return static_cast<T>(static_cast<int64_t>(res));
      }
      return SafeFloatToInteger<T>(static_cast<double>(t_val) + d_bias);
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
      if (is_int_bias) {
        const int64_t b = static_cast<int64_t>(d_bias);
        const uint64_t u_val = static_cast<uint64_t>(static_cast<int64_t>(t_val));
        const uint64_t res = (b >= 0) ? (u_val - static_cast<uint64_t>(b))
                                      : (u_val + static_cast<uint64_t>(-b));
        return static_cast<T>(static_cast<int64_t>(res));
      }
      return SafeFloatToInteger<T>(static_cast<double>(t_val) - d_bias);
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

    const double d_bias = static_cast<double>(bias);
    const bool is_int_bias = (d_bias > -kMaxUnsigned && d_bias < kMaxUnsigned && d_bias == std::floor(d_bias));
    if (c1) {
      if (is_int_bias) {
        const uint64_t u_val = static_cast<uint64_t>(t_val);
        const uint64_t res = (d_bias >= 0.0) ? (u_val + static_cast<uint64_t>(d_bias))
                                             : (u_val - static_cast<uint64_t>(-d_bias));
        return static_cast<T>(res);
      }
      return SafeFloatToInteger<T>(static_cast<double>(t_val) + d_bias);
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
      if (is_int_bias) {
        const uint64_t u_val = static_cast<uint64_t>(t_val);
        const uint64_t res = (d_bias >= 0.0) ? (u_val - static_cast<uint64_t>(d_bias))
                                             : (u_val + static_cast<uint64_t>(-d_bias));
        return static_cast<T>(res);
      }
      return SafeFloatToInteger<T>(static_cast<double>(t_val) - d_bias);
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
