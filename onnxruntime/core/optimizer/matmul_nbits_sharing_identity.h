// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/framework/tensor.h"

namespace onnxruntime {

// Computes a stable "sharing identity" for a MatMulNBits weight group synthesized by a graph
// transformer (the transposed/packed quantized B weight, its scales, and optional zero points).
//
// The identity is a pure function of the source-derived tensor contents and shapes that fully
// determine the MatMulNBits compute semantics. Consequently:
//   - The same logical weight in two sessions of the same model yields the same identity, enabling
//     the pre-packed weight buffer to be shared across sessions.
//   - Any difference that changes the result (different quantized weight, scales, or zero points)
//     yields a different identity, so distinct weights never collide.
//
// This makes cross-session pre-packed weight sharing correct by construction, independent of whether
// the kernel's packed-byte representation happens to capture every semantic input. SessionState
// keys the shared pre-packed weights container on this identity (see
// Graph::AddSharedInitializerIdentity / SessionState::PrepackConstantInitializedTensors).
std::string ComputeMatMulNBitsSharingIdentity(const Tensor& weight,
                                              const Tensor& scale,
                                              const Tensor* zero_point);

}  // namespace onnxruntime
