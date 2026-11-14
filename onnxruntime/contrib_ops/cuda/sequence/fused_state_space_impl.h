// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace onnxruntime::contrib::cuda {

// Shared production normalization launch; also permits small normalization-only kernel regressions.
cudaError_t LaunchMamba2VerifyNormalization(cudaStream_t stream, int rows, int width, int groups, int projection_stride,
                                            float epsilon, const void* projection, const float* readout,
                                            const float* norm_weight, void* output);

// State snapshots use [rows, heads, state_size, head_size], not [rows, heads, head_size, state_size].
// An invalid device selector produces NaN computed rows without reading either input state.
cudaError_t LaunchMamba2Verify(cudaStream_t stream, int rows, int64_t slots, int heads, int groups, float epsilon,
                               float time_step_min, const void* projection, const void* conv_state,
                               const float* ssm_state, const void* conv_weight, const void* conv_bias,
                               const float* dt_bias, const float* a_negative, const float* d_skip,
                               const float* norm_weight, const int32_t* state_row, void* output, void* conv_snapshots,
                               float* ssm_snapshots, void* xbc, float* readout);

// Packed multi-sequence Mamba2Verify over banked recurrent state [num_banks, slots, ...].
cudaError_t LaunchMamba2VerifyBatched(cudaStream_t stream, int total_rows, int batch_size, int64_t num_banks,
                                      int64_t slots, int heads, int groups, float epsilon, float time_step_min,
                                      const void* projection, const void* conv_state, const float* ssm_state,
                                      const void* conv_weight, const void* conv_bias, const float* dt_bias,
                                      const float* a_negative, const float* d_skip, const float* norm_weight,
                                      const int32_t* state_row, const int32_t* cu_seqlens,
                                      const int32_t* state_slots, void* output, void* conv_snapshots,
                                      float* ssm_snapshots, void* xbc, float* readout);
}  // namespace onnxruntime::contrib::cuda
