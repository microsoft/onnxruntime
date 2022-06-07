// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

// contains tag types to determine how to compute Gelu
namespace gelu_computation_mode {

// use approximation
struct Approximation {};
// do not use approximation
struct Default {};

}  // namespace gelu_computation_mode

}  // namespace onnxruntime
