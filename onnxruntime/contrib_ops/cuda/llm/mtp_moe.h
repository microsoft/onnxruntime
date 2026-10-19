// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace mtp_moe {

constexpr int kHiddenSize = 2560;
constexpr int kExpertCount = 512;
constexpr int kTopK = 10;
constexpr int kFfnSize = 640;
constexpr int kScratchStride = 32 + kTopK * kFfnSize;
constexpr int kMaxRows = 65535;

inline Status ValidateRowCount(int64_t rows) {
  if (rows <= 0 || rows > kMaxRows) {
    return Status(common::StatusCategory::ONNXRUNTIME, common::StatusCode::INVALID_ARGUMENT,
                  "MtpMoe requires between 1 and 65535 rows");
  }
  return Status::OK();
}

Status LaunchRouter(const __nv_bfloat16* router_logits, float* scratch,
                    int rows, cudaStream_t stream);

Status LaunchNvfp4MtpMoe(const __nv_bfloat16* input, const __nv_bfloat16* router_logits,
                         const uint8_t* gate_up_weights, const uint8_t* down_weights,
                         float gate_up_global_scale, float down_global_scale,
                         float* scratch, __nv_bfloat16* output,
                         int rows, cudaStream_t stream);

}  // namespace mtp_moe
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
