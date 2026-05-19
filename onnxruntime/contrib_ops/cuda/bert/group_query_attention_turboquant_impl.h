// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/framework/allocator.h"
#include "contrib_ops/cuda/bert/attention_data.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Launches the TurboQuant CUDA path for GroupQueryAttention.
//
// Routing:
//   - When parameters.is_first_prompt: encodes incoming K/V into the present cache
//     (TurboQuant slot layout) AND runs fp16 attention output (using non-encoded inputs).
//   - When sequence_length == 1 (decode): encodes the single new K/V token and appends
//     to present cache, then computes attention scores in rotated space against the
//     packed cache via TQDecodeScoreKernel, applies softmax, and computes the V
//     weighted sum via TQDecodeWeightedSumKernel.
//
// Only `(MLFloat16, uint8_t)` and `(BFloat16, uint8_t)` (T, U) instantiations are valid.
//
// Returns Status::OK on success, or an error if (head_size, key_bits, value_bits)
// is unsupported (currently head_size in {64, 128, 256}, key_bits in {3, 4},
// value_bits in {3, 4}).
template <typename T, typename U>
Status LaunchTurboQuantAttention(
    const cudaDeviceProp& device_prop,
    Stream* stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data);

// Standalone wrapper for unit tests: takes raw K and V, encodes them to TurboQuant
// format, decodes them back to fp16, and writes the result to k_recon / v_recon.
//
// This exercises the encode/decode kernels without the full attention pipeline.
// Used by gtests in test/contrib_ops/turboquant_kv_test.cc.
template <typename T>
Status LaunchTurboQuantEncodeDecodeRoundtrip(
    const cudaDeviceProp& device_prop,
    Stream* stream,
    int batch_size,
    int n_kv_heads,
    int seq_len,
    int head_size,
    int key_bits,       // 3 or 4
    int value_bits,     // 3 or 4
    bool norm_correction,
    const T* K,         // (B, H_kv, S, D) input fp16/bf16 keys
    const T* V,         // (B, H_kv, S, D) input fp16/bf16 values
    const T* k_codebook,    // (2^key_bits) static centroids
    const T* hadamard,      // (D, D) Walsh-Hadamard matrix (unused — we apply FWHT in kernel)
    T* k_recon,         // (B, H_kv, S, D) output: K reconstructed in rotated space
    T* v_recon          // (B, H_kv, S, D) output: V reconstructed (uniform dequant)
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
