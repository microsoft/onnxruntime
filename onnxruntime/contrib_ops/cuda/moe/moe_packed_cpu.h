// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CPU GGUF cold-expert kernels shared by the MoEPacked CPU/GPU offload decode path.

#pragma once

#include <cstddef>
#include <cstdint>

#if (defined(_M_AMD64) && !defined(_M_ARM64EC)) || defined(__x86_64__)
#define ORT_MOE_PACKED_CPU_X86_64 1
#else
#define ORT_MOE_PACKED_CPU_X86_64 0
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace moe_packed_cpu {

#if ORT_MOE_PACKED_CPU_X86_64
// AVX2 + FMA + F16C kernels implemented in moe_packed_cpu_avx2.cc. Call them only after
// qmoe_shared::CpuHasAvx2FmaF16c() returns true. Each returns false, without writing outputs, when `type` has no AVX2
// kernel so the caller can use the scalar path. K must be a multiple of the GGUF block size of `type`.

// dots[r] = dot(row, x_rows[r]) for r in [0, R).
bool GgufDotRowAvx2(const uint8_t* row, int type, int K, const float* const* x_rows, int R, float* dots);

// Gate and up rows of the same type: gate_dots[r] = dot(gate_row, x_rows[r]), up_dots[r] = dot(up_row, x_rows[r]).
bool GgufDualDotRowAvx2(const uint8_t* gate_row, const uint8_t* up_row, int type, int K,
                        const float* const* x_rows, int R, float* gate_dots, float* up_dots);

// Two gate/up channel pairs of the same type, evaluated with register accumulators only.
bool GgufDualDot2ChAvx2(const uint8_t* gate_row0, const uint8_t* up_row0, const uint8_t* gate_row1,
                        const uint8_t* up_row1, int type, int K, const float* const* x_rows, int R,
                        float* gate_dots0, float* up_dots0, float* gate_dots1, float* up_dots1);

// Rows [0, n) starting at row0 with a stride of row_bytes, where n is n_rows rounded down to a multiple of 4:
// dots[r * dots_ld + i] = dot(row_i, x_rows[r]). Returns n, or 0 when `type` has no AVX2 kernel.
int GgufDotRowsBy4Avx2(const uint8_t* row0, size_t row_bytes, int n_rows, int type, int K,
                       const float* const* x_rows, int R, float* dots, int dots_ld);
#endif

}  // namespace moe_packed_cpu
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
