// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include "core/framework/tensor_shape.h"
#include "core/common/status.h"
#include "core/common/narrow.h"
#include "core/common/inlined_containers.h"
#include "core/providers/cpu/nn/layer_norm_macro.h"

namespace onnxruntime {

constexpr const char* kLayerNormInputShapeMismatchError =
    "Scale and (optional) bias must match X.shape[axis:] or be NumPy-broadcastable to it.";

constexpr const char* kLayerNormInvalidSize = "Size of X.shape[axis:] must be at least 1, got ";

constexpr int64_t kLayerNormInvalidInput = -1;

struct LayerNormParams {
  int64_t num_rows;
  int64_t norm_size;  // size per row
  int64_t scale_size;
  int64_t bias_size;
  int64_t broadcast_param;
  bool use_generic_broadcast{false};  // true: full NumPy-style broadcast; false: legacy broadcast_param path
  onnxruntime::InlinedVector<int64_t, 8> x_dims;
  onnxruntime::InlinedVector<int64_t, 8> x_inner_dims;  // X.shape[axis:]
  onnxruntime::InlinedVector<int64_t, 8> scale_dims;
  onnxruntime::InlinedVector<int64_t, 8> bias_dims;
  onnxruntime::InlinedVector<int64_t, 8> scale_strides;
  onnxruntime::InlinedVector<int64_t, 8> bias_strides;
  int64_t axis{0};
  int64_t last_rank{0};
  onnxruntime::InlinedVector<int64_t, 8> scale_inner_inc;  // scale strides for inner dims [axis..]
  onnxruntime::InlinedVector<int64_t, 8> bias_inner_inc;   // bias  strides for inner dims [axis..]
  onnxruntime::InlinedVector<int64_t, 8> x_outer_strides;  // X strides for outer dims [0..axis-1]
};

// Fast-path broadcasting for axis = 2:
// When X shape is (B, S, ...), and x_row (index of one row in X) is in the range of [0, B * S).
// We support the following scale/bias shapes in this path:
//    When scale and bias shape is (1, 1, ...) or (...), value of broadcast_param is 0.
//    When scale and bias shape is (B, 1, ...), value of broadcast_param is S.
//    When scale and bias shape is (B, S, ...), value of broadcast_param is 1.
//    When scale and bias shape is (1, S, ...), value of broadcast_param is -S.
// For all other NumPy-broadcastable shapes we fall back to the generic
// broadcasting path (use_generic_broadcast = true) and ignore broadcast_param.

class LayerNormHelper {
 public:
  static Status CheckInputs(const TensorShape& x_shape,
                            const TensorShape& scale_shape,
                            const TensorShape& bias_shape,
                            bool has_bias,
                            int64_t axis,
                            LayerNormParams& params) {
    // Initialize basic layout parameters: how many rows we have and how many elements
    // are normalized per row, as well as the total scale/bias sizes.
    params.num_rows = x_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis));
    params.norm_size = x_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis));
    params.scale_size = scale_shape.Size();
    params.bias_size = has_bias ? bias_shape.Size() : 0;

    params.broadcast_param = 0;
    params.axis = axis;

    // Allow norm_size == 1 (scalar normalization is valid according to ONNX spec).
    if (params.norm_size < 1) {
      params.broadcast_param = kLayerNormInvalidInput;
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, kLayerNormInvalidSize, params.norm_size);
    } else if (params.scale_size != params.norm_size || (has_bias && params.bias_size != params.scale_size)) {
      params.broadcast_param = GetBroadcastParam(x_shape, scale_shape, has_bias ? &bias_shape : nullptr, axis);
      // Try to encode simple (B, S, ...) layouts into broadcast_param so that the
      // fast-path can be used. If this fails, broadcast_param will be set to
      // kLayerNormInvalidInput and we may fall back to generic broadcasting later.
    }
    const size_t xr = x_shape.NumDimensions();
    const size_t sr = scale_shape.NumDimensions();
    const size_t br = has_bias ? bias_shape.NumDimensions() : 0;

    if (sr > xr || (has_bias && br > xr)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             kLayerNormInputShapeMismatchError,
                             " Scale/Bias rank cannot exceed Input rank.");
    }

    params.x_dims.clear();
    params.x_dims.reserve(xr);
    for (size_t i = 0; i < xr; ++i) {
      params.x_dims.push_back(x_shape.GetDims()[i]);
    }

    // Right-align scale and bias shapes
    params.scale_dims.clear();
    params.scale_dims.resize(xr, 1);
    for (size_t i = 0; i < sr; ++i) {
      params.scale_dims[xr - 1 - i] = scale_shape.GetDims()[sr - 1 - i];
    }

    params.bias_dims.clear();
    if (has_bias) {
      params.bias_dims.resize(xr, 1);
      for (size_t i = 0; i < br; ++i) {
        params.bias_dims[xr - 1 - i] = bias_shape.GetDims()[br - 1 - i];
      }
    }

    // Validate broadcastability
    const bool sc_ok = IsNumpyBroadcastable(params.scale_dims, params.x_dims);
    const bool bi_ok = !has_bias || IsNumpyBroadcastable(params.bias_dims, params.x_dims);
    if (!sc_ok || !bi_ok) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             kLayerNormInputShapeMismatchError,
                             " X.shape=", x_shape,
                             " scale.shape=", scale_shape,
                             " bias.shape=", bias_shape,
                             " and axis=", axis);
    }

    // Compute strides for scale/bias once
    params.scale_strides = MakeStrides(params.scale_dims);
    params.bias_strides.clear();
    if (has_bias) {
      params.bias_strides = MakeStrides(params.bias_dims);
    }

    // Detect dependency on outer dimensions [0..axis-1]
    bool outer_dep = false;
    for (int64_t i = 0; i < axis; ++i) {
      const size_t idx = static_cast<size_t>(i);
      if (params.scale_strides[idx] != 0 ||
          (has_bias && params.bias_strides[idx] != 0)) {
        outer_dep = true;
        break;
      }
    }

    // Decide if we need the generic NumPy-style broadcasting path
    params.use_generic_broadcast = outer_dep || (params.broadcast_param == kLayerNormInvalidInput);

    if (params.use_generic_broadcast) {
      // Cache inner dims X.shape[axis:]
      params.last_rank = onnxruntime::narrow<int64_t>(xr) - axis;
      params.x_inner_dims.clear();
      params.x_inner_dims.reserve(params.last_rank > 0 ? static_cast<size_t>(params.last_rank) : 0);
      for (size_t i = static_cast<size_t>(axis); i < xr; ++i) {
        params.x_inner_dims.push_back(params.x_dims[i]);
      }

      // Precompute inner increments for scale/bias over [axis..]
      params.scale_inner_inc.clear();
      params.bias_inner_inc.clear();
      for (size_t i = static_cast<size_t>(axis); i < xr; ++i) {
        params.scale_inner_inc.push_back(params.scale_strides[i]);
        if (has_bias) {
          params.bias_inner_inc.push_back(params.bias_strides[i]);
        }
      }

      // X outer strides [0..axis-1], used only in generic path
      params.x_outer_strides.clear();
      params.x_outer_strides.resize(static_cast<size_t>(axis), 1);
      if (axis > 1) {
        for (int64_t d = axis - 2; d >= 0; --d) {
          const size_t du = static_cast<size_t>(d);
          params.x_outer_strides[du] =
              params.x_outer_strides[du + 1] * params.x_dims[du + 1];
        }
      }
    } else {
      // Fast-path: we don't need inner/outer increments
      params.last_rank = 0;
      params.x_inner_dims.clear();
      params.scale_inner_inc.clear();
      params.bias_inner_inc.clear();
      params.x_outer_strides.clear();
    }

    return Status::OK();
  }

 private:
  static bool IsNumpyBroadcastable(gsl::span<const int64_t> a,
                                   gsl::span<const int64_t> b) {
    ORT_ENFORCE(a.size() == b.size());
    for (size_t k = 0; k < a.size(); ++k) {
      const int64_t ak = a[k];
      const int64_t bk = b[k];
      if (!(ak == 1 || ak == bk)) {
        return false;
      }
    }
    return true;
  }
  static InlinedVector<int64_t, 8> MakeStrides(const InlinedVector<int64_t, 8>& dims) {
    InlinedVector<int64_t, 8> strides(dims.size(), 0);
    if (dims.empty()) return strides;

    int64_t running = 1;
    for (ptrdiff_t i = dims.size() - 1; i >= 0; --i) {
      size_t idx = static_cast<size_t>(i);
      strides[idx] = (dims[idx] == 1) ? 0 : running;
      running *= std::max<int64_t>(1, dims[idx]);
    }

    return strides;
  }

  static int64_t GetBroadcastParam(const TensorShape& x_shape,
                                   const TensorShape& scale_shape,
                                   const TensorShape* bias_shape,
                                   int64_t axis) {
    // Note that when size of scale and bias is norm_size, it won't enter this function (see CheckInputs).

    // X shape is (B, S, ...)
    if (axis == 2 &&
        x_shape.NumDimensions() >= 3 &&
        x_shape.NumDimensions() == scale_shape.NumDimensions() &&
        (bias_shape == nullptr || *bias_shape == scale_shape)) {
      for (size_t i = 2; i < x_shape.NumDimensions(); ++i) {
        if (x_shape.GetDims()[i] != scale_shape.GetDims()[i]) {
          // scale cannot be broadcasted to X. It is invalid input.
          return kLayerNormInvalidInput;
        }
      }

      if (x_shape.GetDims()[0] == scale_shape.GetDims()[0]) {
        // scale and bias shape is (B, S, ...).
        if (x_shape.GetDims()[1] == scale_shape.GetDims()[1]) {
          return 1;
        }

        // scale and bias shape is (B, 1, ...), returns S
        if (scale_shape.GetDims()[1] == 1) {
          return x_shape.GetDims()[1];
        }
      } else if (scale_shape.GetDims()[0] == 1) {
        // scale and bias shape is (1, S, ...), returns -S
        if (x_shape.GetDims()[1] == scale_shape.GetDims()[1]) {
          return -(x_shape.GetDims()[1]);
        }
      }
    }

    // Other cases that are not supported.
    return kLayerNormInvalidInput;
  }
};

}  // namespace onnxruntime
