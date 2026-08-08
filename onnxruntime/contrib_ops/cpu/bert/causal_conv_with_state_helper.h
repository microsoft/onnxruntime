// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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

// Derives the expected past_state / present_state shape and validates past_state against it.
// `state_length` is the carry length along the causal axis, i.e. kernel_size - 1.
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
                   TensorShape& state_shape) {
  state_shape = state_window > 0
                    ? TensorShape({state_window, batch_size, channels, state_length})
                    : TensorShape({batch_size, channels, state_length});

  if (past_state != nullptr && past_state->Shape() != state_shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_state' is expected to have shape ", state_shape.ToString(),
                           ", got ", past_state->Shape().ToString(),
                           ". CausalConvWithState uses (batch_size, channels, kernel_size - 1) when "
                           "the state_window attribute is absent or 0, and "
                           "(state_window, batch_size, channels, kernel_size - 1) otherwise.");
  }

  return Status::OK();
}

}  // namespace causal_conv_with_state_helper
}  // namespace contrib
}  // namespace onnxruntime
