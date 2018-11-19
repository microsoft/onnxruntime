// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace detail {

// Helper struct for an activation function call information
template <typename TFunc>
struct ActivationInfo {
  TFunc func;
  float alpha;
  float beta;
};

}  // namespace detail
}  // namespace contrib
}  // namespace onnxruntime
