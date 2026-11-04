// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {

// Boundary-safe diagnostic types shared by native execution and provider DLLs.
enum class MatMulNBitsRuntimeRoute : uint32_t {
  EmptyOutput,
  PackedLut,
  PackedQNBit,
  Unpacked,
  CudaFused,
  CudaFpAIntBGemv,
  CudaFpAIntBGemm,
  CudaMixedFpAIntB,
  CudaDequantizeGemm,
};

struct MatMulNBitsCudaRouteMetadata {
  MatMulNBitsRuntimeRoute route;
  int64_t reconstruction_k;
};

}  // namespace onnxruntime
