// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <functional>
#include "core/common/safeint.h"

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/tensor.h"
#endif

namespace onnxruntime {

/**
Returns whether `axis` is in range [-`tensor_rank`, `tensor_rank`).
**/
constexpr inline bool IsAxisInRange(int64_t axis, int64_t tensor_rank) {
  return axis >= -tensor_rank && axis <= tensor_rank - 1;
}

/**
Handle a potentially negative axis. Enforces negative axis is valid.
@param axis Axis to convert from negative to positive if needed.
@param tensor_rank Rank of tensor axis applies to. Tensor::Shape()::NumDimensions().
@returns non-negative axis.
*/
inline int64_t HandleNegativeAxis(int64_t axis, int64_t tensor_rank) {
  ORT_ENFORCE(IsAxisInRange(axis, tensor_rank), "axis ", axis,
              " is not in valid range [-", tensor_rank, ",", tensor_rank - 1, "]");
  // Handle negative axis
  return axis < 0 ? axis + tensor_rank : axis;
}

/**
Returns true if given tensor is a scalar or 1D tensor of size 1
**/
inline bool IsScalarOr1ElementVector(const Tensor* input) {
  if (input->Shape().NumDimensions() == 0 ||
      (input->Shape().NumDimensions() == 1 && input->Shape().Size() == 1)) {
    return true;
  } else {
    return false;
  }
}

/**
Clamps input between provided min and max values
**/
inline float clamp(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

inline AutoPadType StringToAutoPadType(const std::string& str) {
  if (str.empty()) {
    return AutoPadType::NOTSET;
  }
  if (str == "NOTSET") {  // in onnx spec, default value is "NOTSET"
    return AutoPadType::NOTSET;
  }
  if (str == "VALID") {
    return AutoPadType::VALID;
  }
  if (str == "SAME_UPPER") {
    return AutoPadType::SAME_UPPER;
  }
  if (str == "SAME_LOWER") {
    return AutoPadType::SAME_LOWER;
  }
  ORT_ENFORCE(false, "Unknown AutoPadType String");
}

// helper function

constexpr inline int64_t ComputeOutputShape(const int64_t in_dim,
                                            const int64_t stride, const int64_t kernel, const int64_t dilation,
                                            const int64_t pad_head, const int64_t pad_tail) {
  const SafeInt<int64_t> dkernel = SafeInt<int64_t>(dilation) * (kernel - 1) + 1;
  int64_t dkernel_value = SafeInt<int64_t>(in_dim) + pad_head + pad_tail - dkernel;
  return static_cast<int64_t>(static_cast<double>(dkernel_value) / stride + 1);
}

inline Status ComputePad(const int64_t in_dim,
                         const int64_t stride, const int64_t kernel, const int64_t dilation,
                         AutoPadType pad_type,
                         int64_t& pad_head, int64_t& pad_tail,
                         bool force_symmetric_auto_padding = false) {
  switch (pad_type) {
    case AutoPadType::NOTSET:
      break;
    case AutoPadType::VALID: {
      pad_head = 0;
      pad_tail = 0;
    } break;
    case AutoPadType::SAME_UPPER:
    case AutoPadType::SAME_LOWER: {
      if (1 != dilation)
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                      "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.");

      // The ONNX spec says if `auto_pad` attribute is set, pad until the `legacy_target_size`
      // is `ceil (in_dim / stride)`. The following line of code is essentially just that and
      // is retained as is
      SafeInt<int64_t> legacy_target_size = (SafeInt<int64_t>(in_dim) + stride - 1) / stride;
      SafeInt<int64_t> pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
      // out_dim = floor((in_dim + 2p - k) / s) + 1
      // => if (in_dim + 2p - k) is not divisible by s we can remove the floor with following equation:
      // out_dim + eps = ((in_dim + 2p - k) / s) + 1 ;where eps is in [0.0, 1.0]
      // therefore in edge cases padding can lower calculated above than it should be
      SafeInt<int64_t> actual_out_size = ComputeOutputShape(in_dim, stride, kernel, /*dilation*/ 1,
                                                            pad_needed, pad_needed);
      if (actual_out_size < legacy_target_size) {
        pad_needed += 1;
      }
      // make sure padding is symmetric
      if (force_symmetric_auto_padding) {
        // Inlining math::roundUpPow2() from util/math.h to avoid bringing in the transitive dependencies.
        pad_needed = (pad_needed + 1) & ~1;
      }

      if (pad_type == AutoPadType::SAME_LOWER)
        pad_head = (pad_needed + 1) / 2;
      else
        pad_head = pad_needed / 2;

      pad_tail = pad_needed - pad_head;
    } break;
    default:
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "ComputePad: pad type not supported.");
  }

  return Status::OK();
}

inline Status ComputePadAndOutputShape(const int64_t in_dim,
                                       const int64_t stride, const int64_t kernel, const int64_t dilation,
                                       AutoPadType pad_type,
                                       int64_t& pad_head, int64_t& pad_tail,
                                       int64_t& out_dim,
                                       bool force_symmetric_auto_padding = false) {
  ORT_RETURN_IF_ERROR(
      ComputePad(in_dim, stride, kernel, dilation, pad_type, pad_head, pad_tail, force_symmetric_auto_padding));
  out_dim = ComputeOutputShape(in_dim, stride, kernel, dilation, pad_head, pad_tail);
  return Status::OK();
}

constexpr inline int64_t ComputeTotalPad(int64_t in_size, int64_t stride, int64_t adj,
                                         int64_t kernel, int64_t dilation, int64_t out_size) {
  return std::max<int64_t>(0, (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1 - out_size);
}

inline void DistributePadding(AutoPadType pad_type, const int64_t& total_pad,
                              int64_t& pad_head, int64_t& pad_tail) {
  if (pad_type == AutoPadType::SAME_UPPER) {
    // pad more on tail when total_pad is odd.
    pad_head = total_pad / 2;
    pad_tail = total_pad - total_pad / 2;
  } else {
    // When pad_type is NOTSET, SAME_LOWER or VALID,
    // pad more on head when total_pad is odd.
    pad_head = total_pad - total_pad / 2;
    pad_tail = total_pad / 2;
  }
}

// Note: This helper function will not have overflow protection
template <template <typename...> class Container, typename T>
T Product(const Container<T>& c) {
  return accumulate(c.cbegin(), c.cend(), static_cast<T>(1), std::multiplies<T>());
}

/// <summary>
/// Compute the output shape for broadcasting the given input shapes of lhs and rhs.
/// </summary>
inline Status ComputeBroadcastOutputShape(const std::string& node_name,
                                          const TensorShape& lhs_shape,
                                          const TensorShape& rhs_shape,
                                          TensorShape& out_shape) {
  size_t lhs_rank = lhs_shape.NumDimensions();
  size_t rhs_rank = rhs_shape.NumDimensions();
  size_t out_rank = std::max(lhs_rank, rhs_rank);

  std::vector<int64_t> output_dims(out_rank, 0);
  for (size_t i = 0; i < out_rank; ++i) {
    int64_t lhs_dim = 1;
    if (i < lhs_rank)
      lhs_dim = lhs_shape[lhs_rank - 1 - i];
    int64_t rhs_dim = 1;
    if (i < rhs_rank)
      rhs_dim = rhs_shape[rhs_rank - 1 - i];
    int64_t max = std::max(lhs_dim, rhs_dim);
    int64_t min = std::min(lhs_dim, rhs_dim);
    int64_t out_dim = (min == 0 ? min : max);  // special case a dim value of 0.
    if (lhs_dim != out_dim && lhs_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": left operand cannot broadcast on dim ", lhs_rank - 1 - i,
                             " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    if (rhs_dim != out_dim && rhs_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": right operand cannot broadcast on dim ", rhs_rank - 1 - i,
                             " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    output_dims[out_rank - 1 - i] = out_dim;
  }
  out_shape = TensorShape(output_dims);
  return Status::OK();
}

}  // namespace onnxruntime
