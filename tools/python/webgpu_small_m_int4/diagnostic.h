// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdlib>
#include <initializer_list>
#include <string>
#include <string_view>

#include "contrib_ops/webgpu/quantization/matmul_nbits_small_m_simt.h"

namespace onnxruntime::contrib::webgpu {
namespace {

uint32_t DiagnosticTileSize(const char* name, uint32_t fallback, std::initializer_list<uint32_t> supported) {
  const char* text = std::getenv(name);
  if (text == nullptr) {
    return fallback;
  }
  for (uint32_t value : supported) {
    if (std::string_view{text} == std::to_string(value)) {
      return value;
    }
  }
  ORT_THROW("Unsupported diagnostic tile setting: ", name, "=", text);
}

Status RunDiagnosticSmallMSimt(const Tensor* a, const Tensor* b, const Tensor* scales,
                               const Tensor* zero_points, const Tensor* bias, uint32_t M, uint32_t N,
                               uint32_t K, uint32_t block_size, uint32_t batch_count, uint32_t weight_index,
                               const Tensor* weight_index_indirect, uint32_t dispatch_M, uint32_t nbits,
                               onnxruntime::webgpu::ComputeContext& context, Tensor* y) {
  return RunSmallMSimt(a, b, scales, zero_points, bias, M, N, K, block_size, batch_count, weight_index,
                       weight_index_indirect, dispatch_M, nbits, context, y,
                       DiagnosticTileSize("ORT_WEBGPU_SMALL_M_BN", 16, {8, 16, 32}),
                       DiagnosticTileSize("ORT_WEBGPU_SMALL_M_BK", 512, {256, 512, 1024}),
                       DiagnosticTileSize("ORT_WEBGPU_SMALL_M_A_BLOCK_PADDING", 0, {0, 1}));
}

}  // namespace
}  // namespace onnxruntime::contrib::webgpu
