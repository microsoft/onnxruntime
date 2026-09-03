// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {
namespace causal_conv_with_state_helper {

// Reads and validates the optional `state_window` attribute.
//
// 0 (the default, i.e. attribute absent) selects the legacy unwindowed state layout. Every model
// exported before the attribute existed lands here, so this must stay the default.
template <typename TKernelInfo>
Status ParseStateWindow(const TKernelInfo& info, int& state_window) {
  const int64_t value = info.template GetAttrOrDefault<int64_t>("state_window", 0);
  if (value < 0 || value > kMaxStateWindow) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "state_window must be in [0, ", kMaxStateWindow, "], got ", value);
  }
  state_window = static_cast<int>(value);
  return Status::OK();
}

// Reads and validates the optional `dilation` attribute.
//
// 1 (the default, i.e. attribute absent) is the undilated case, which is what every model exported
// before the attribute existed uses, so this must stay the default.
template <typename TKernelInfo>
Status ParseDilation(const TKernelInfo& info, int& dilation) {
  const int64_t value = info.template GetAttrOrDefault<int64_t>("dilation", 1);
  if (value < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "dilation must be >= 1, got ", value);
  }
  dilation = static_cast<int>(value);
  return Status::OK();
}

// Reads and validates the optional `channels_last` attribute.
//
// 0 (the default, i.e. attribute absent) is the channels-first (batch_size, channels, seq_len)
// layout that every model exported before the attribute existed uses.
template <typename TKernelInfo>
Status ParseChannelsLast(const TKernelInfo& info, bool& channels_last) {
  const int64_t value = info.template GetAttrOrDefault<int64_t>("channels_last", 0);
  if (value != 0 && value != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "channels_last must be 0 or 1, got ",
                           value);
  }
  channels_last = (value == 1);
  return Status::OK();
}

// Element strides of a (batch, position, channel) view over an activation or state tensor.
// Both supported layouts are dense, so a single strided view covers them and every kernel can be
// written once against (batch, position, channel) coordinates.
struct Layout {
  int64_t batch_stride;
  int64_t pos_stride;
  int64_t chan_stride;

  int64_t Offset(int64_t b, int64_t pos, int64_t c) const {
    return b * batch_stride + pos * pos_stride + c * chan_stride;
  }
};

// `length` is the extent of the position axis: seq_len for activations, state_length for state.
constexpr Layout MakeLayout(bool channels_last, int64_t channels, int64_t length) {
  return channels_last ? Layout{channels * length, channels, 1}
                       : Layout{channels * length, 1, length};
}

// Derives the expected channels-last past_state / present_state shape,
// (batch_size, state_length, d_1, ..., d_n), from the input shape and validates past_state.
// The trailing channel axes are copied verbatim from the input, so a caller that keeps
// hyper-connections and hidden size as separate axes gets the same split back and needs no reshape.
template <typename T>
Status CheckInputsChannelsLast(int state_window,
                               const TensorShape& input_shape,
                               int state_length,
                               const T* past_state,
                               TensorShape& state_shape,
                               std::string_view op_name) {
  TensorShapeVector dims;
  if (state_window > 0) {
    dims.push_back(state_window);
  }
  dims.push_back(input_shape[0]);
  dims.push_back(state_length);
  for (size_t i = 2; i < input_shape.NumDimensions(); ++i) {
    dims.push_back(input_shape[i]);
  }
  state_shape = TensorShape(dims);

  if (past_state != nullptr && past_state->Shape() != state_shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_state' is expected to have shape ", state_shape.ToString(),
                           ", got ", past_state->Shape().ToString(),
                           ". ", op_name,
                           " with channels_last = 1 uses "
                           "(batch_size, (kernel_size - 1) * dilation, d_1, ..., d_n), optionally "
                           "led by a state_window axis.");
  }

  return Status::OK();
}

// Derives the expected past_state / present_state shape and validates past_state against it.
// `state_length` is the carry length along the causal axis, i.e. (kernel_size - 1) * dilation.
//
// state_window == 0 -> (batch_size, channels, state_length). A single state with no window axis:
// the backward-compatible layout that models exported before the attribute existed use.
//
// state_window == W > 0 -> (W, batch_size, channels, state_length). The window holds the carry
// state after each of the last W positions, right-aligned: slot j is the state after position
// (seq_len - W + j), so slot W-1 is the state after the last position and is the only slot read
// back as past_state. W == 1 is therefore the legacy layout with a leading unit axis.
//
// The window axis leads the batch axis so that each slot is one contiguous
// (batch_size, channels, state_length) block. That keeps "fetch/replace the last state" a single
// contiguous range for any batch size, which is what a speculative decoder needs when it crops the
// state back to an accepted prefix.
template <typename T>
Status CheckInputs(int state_window,
                   int batch_size,
                   int channels,
                   int state_length,
                   const T* past_state,
                   TensorShape& state_shape,
                   std::string_view op_name) {
  state_shape = state_window > 0
                    ? TensorShape({state_window, batch_size, channels, state_length})
                    : TensorShape({batch_size, channels, state_length});

  if (past_state != nullptr && past_state->Shape() != state_shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_state' is expected to have shape ", state_shape.ToString(),
                           ", got ", past_state->Shape().ToString(),
                           ". ", op_name,
                           " uses (batch_size, channels, (kernel_size - 1) * dilation) when "
                           "the state_window attribute is absent or 0, and "
                           "(state_window, batch_size, channels, (kernel_size - 1) * dilation) "
                           "otherwise.");
  }

  return Status::OK();
}

}  // namespace causal_conv_with_state_helper
}  // namespace contrib
}  // namespace onnxruntime
