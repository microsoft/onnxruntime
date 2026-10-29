// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_int2_prefill.h"
#include "cutlass/epilogue/thread/activation.h"
#include "contrib_ops/cuda/llm/cutlass_type_conversion.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_int2.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_util_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_activation_kernels.cuh"
#include "core/common/safeint.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {
namespace {

struct PrefillWorkspace {
  size_t bytes = 0;
  size_t permuted_rows;
  size_t permuted_experts;
  size_t expert_offsets;
  size_t counts;
  size_t cumulative_counts;
  size_t blocked_rows;
  size_t expanded_input;
  size_t fc1_output;
  size_t activated_output;
  size_t fc2_output;
  size_t repacked_weights = 0;
  size_t transposed_scales = 0;

  size_t Reserve(size_t count, size_t element_size) {
    const size_t offset = bytes;
    const size_t end = SafeInt<size_t>(bytes) + SafeInt<size_t>(count) * element_size;
    bytes = (SafeInt<size_t>(end) + 255) / 256 * 256;
    return offset;
  }

  explicit PrefillWorkspace(const Int2MoePrefillParams& params) {
    const size_t expanded = SafeInt<size_t>(params.num_rows) * params.top_k;
    const int64_t block_rows = computeNumTokensPerBlock(params.num_rows, params.num_experts);
    const size_t blocks = (params.num_rows + block_rows - 1) / block_rows;
    const size_t count_elements = SafeInt<size_t>(params.num_experts) * blocks;
    permuted_rows = Reserve(expanded, sizeof(int));
    permuted_experts = Reserve(expanded, sizeof(int));
    expert_offsets = Reserve(SafeInt<size_t>(params.num_experts) + 1, sizeof(int64_t));
    counts = Reserve(count_elements, sizeof(int));
    cumulative_counts = Reserve(count_elements, sizeof(int));
    blocked_rows = Reserve(SafeInt<size_t>(params.num_rows) * params.num_experts, sizeof(int));
    expanded_input = Reserve(SafeInt<size_t>(expanded) * params.hidden_size, sizeof(half));
    fc1_output = Reserve(SafeInt<size_t>(expanded) * params.inter_size * 2, sizeof(half));
    activated_output = Reserve(SafeInt<size_t>(expanded) * params.inter_size, sizeof(half));
    fc2_output = expanded_input;
    if (params.weights_are_raw && params.block_size > 0) {
      const size_t fc1_repack_bytes =
          SafeInt<size_t>(params.num_experts) * params.inter_size * params.hidden_size;
      const size_t fc2_repack_bytes =
          SafeInt<size_t>(params.num_experts) * params.hidden_size * (params.inter_size / 2);
      repacked_weights = Reserve(std::max(fc1_repack_bytes, fc2_repack_bytes), sizeof(uint8_t));
      const size_t fc1_scale_elems =
          SafeInt<size_t>(params.num_experts) * (params.inter_size * 2) * (params.hidden_size / params.block_size);
      const size_t fc2_scale_elems =
          SafeInt<size_t>(params.num_experts) * params.hidden_size * (params.inter_size / params.block_size);
      transposed_scales = Reserve(std::max(fc1_scale_elems, fc2_scale_elems), sizeof(half));
    }
  }
};

template <int SrcBits>
__global__ void __launch_bounds__(256) RepackRawIntToSm80Int4ActiveExpertsKernel(
    const uint8_t* __restrict__ raw_weights,
    uint8_t* __restrict__ sm80_int4_weights,
    const int64_t* __restrict__ expert_offsets,
    int n,
    int k) {
  const int e = static_cast<int>(blockIdx.y);
  if (expert_offsets != nullptr && __ldg(&expert_offsets[e + 1]) == __ldg(&expert_offsets[e])) {
    return;
  }
  const int k_half_tiles = k >> 5;  // K / 32
  const int num_uint4_per_expert = n * k_half_tiles;
  const int idx_in_expert = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
  if (idx_in_expert >= num_uint4_per_expert) {
    return;
  }

  const int k_tiles = k >> 6;  // K / 64
  const int k_half = idx_in_expert & 1;
  const int n_low = (idx_in_expert >> 1) & 3;
  const uint32_t idx8 = static_cast<uint32_t>(idx_in_expert >> 3);
  const uint32_t n_high = idx8 / static_cast<uint32_t>(k_tiles);
  const uint32_t k_tile = idx8 - n_high * static_cast<uint32_t>(k_tiles);
  const int row = (static_cast<int>(n_high) << 2) | n_low;
  const int half_k_idx = (static_cast<int>(k_tile) << 1) | k_half;
  const size_t seg_idx = (static_cast<size_t>(e) * n + row) * k_half_tiles + half_k_idx;

  uint32_t pkt[4];
  if constexpr (SrcBits == 2) {
    const uint2 w = __ldg(reinterpret_cast<const uint2*>(raw_weights) + seg_idx);
    pkt[0] = w.x & 0xFFFFu;
    pkt[1] = w.x >> 16;
    pkt[2] = w.y & 0xFFFFu;
    pkt[3] = w.y >> 16;
  } else if constexpr (SrcBits == 3) {
    const uint32_t* src_u32 = reinterpret_cast<const uint32_t*>(raw_weights) + seg_idx * 3;
    const uint32_t w0 = __ldg(&src_u32[0]);
    const uint32_t w1 = __ldg(&src_u32[1]);
    const uint32_t w2 = __ldg(&src_u32[2]);
    pkt[0] = w0 & 0x00FFFFFFu;
    pkt[1] = __funnelshift_r(w0, w1, 24) & 0x00FFFFFFu;
    pkt[2] = __funnelshift_r(w1, w2, 16) & 0x00FFFFFFu;
    pkt[3] = w2 >> 8;
  } else {
    static_assert(SrcBits == 4);
    const uint4 w = __ldg(reinterpret_cast<const uint4*>(raw_weights) + seg_idx);
    pkt[0] = w.x;
    pkt[1] = w.y;
    pkt[2] = w.z;
    pkt[3] = w.w;
  }

  uint32_t out_w[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    if constexpr (SrcBits == 2) {
      uint32_t word = 0x66666666u;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const uint32_t pair4 = (pkt[p] >> (4 * s)) & 0xFu;
        const uint32_t u_even = pair4 & 0x3u;
        const uint32_t u_odd = pair4 >> 2;
        word += (u_even << (4 * p)) | (u_odd << (16 + 4 * p));
      }
      out_w[s] = word;
    } else if constexpr (SrcBits == 3) {
      uint32_t word = 0x44444444u;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const uint32_t pair6 = (pkt[p] >> (6 * s)) & 0x3Fu;
        const uint32_t u_even = pair6 & 0x7u;
        const uint32_t u_odd = pair6 >> 3;
        word += (u_even << (4 * p)) | (u_odd << (16 + 4 * p));
      }
      out_w[s] = word;
    } else {
      uint32_t word = 0u;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const uint32_t pair8 = (pkt[p] >> (8 * s)) & 0xFFu;
        const uint32_t u_even = pair8 & 0xFu;
        const uint32_t u_odd = pair8 >> 4;
        word |= (u_even << (4 * p)) | (u_odd << (16 + 4 * p));
      }
      out_w[s] = word;
    }
  }

  const size_t out_idx = static_cast<size_t>(e) * num_uint4_per_expert + idx_in_expert;
  reinterpret_cast<uint4*>(sm80_int4_weights)[out_idx] = make_uint4(out_w[0], out_w[1], out_w[2], out_w[3]);
}

__device__ __forceinline__ int PrepackInt3SmSlot(int lane, int odd) {
  return ((lane & 2) << 4) | (lane & 28) | ((lane & 1) << 1) | odd;
}

__device__ __forceinline__ uint4 UnpackInt3WordsToSm80Int4(uint32_t w0, uint32_t w1, uint32_t w2) {
  uint32_t pkt[4];
  pkt[0] = w0 & 0x00FFFFFFu;
  pkt[1] = __funnelshift_r(w0, w1, 24) & 0x00FFFFFFu;
  pkt[2] = __funnelshift_r(w1, w2, 16) & 0x00FFFFFFu;
  pkt[3] = w2 >> 8;

  uint32_t out_w[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    uint32_t word = 0x44444444u;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const uint32_t pair6 = (pkt[p] >> (6 * s)) & 0x3Fu;
      const uint32_t u_even = pair6 & 0x7u;
      const uint32_t u_odd = pair6 >> 3;
      word += (u_even << (4 * p)) | (u_odd << (16 + 4 * p));
    }
    out_w[s] = word;
  }
  return make_uint4(out_w[0], out_w[1], out_w[2], out_w[3]);
}

// Each CTA unpacks one (m-block = 16 rows, s_chunk = up to 512 K columns = 16 half_k_idx) tile of prepacked INT3
// weights into SM80 INT4 layout. Reading the 3 KB prepacked tile into conflict-free shared memory first ensures every
// weight byte is fetched once via coalesced uint4 loads and written via coalesced uint4 stores. When k has a single
// 128-col remainder step (e.g. K=640), the last s_chunk CTA also loads and unpacks the remainder using warps 6..7.
template <bool kFoldRem1, bool kHasRemChunk>
__global__ void __launch_bounds__(256) RepackPrepackedInt3ToSm80Int4ActiveExpertsKernel(
    const uint8_t* __restrict__ prepacked_weights,
    uint8_t* __restrict__ sm80_int4_weights,
    const int64_t* __restrict__ expert_offsets,
    int n,
    int k) {
  const int e = static_cast<int>(blockIdx.z);
  if (expert_offsets != nullptr && __ldg(&expert_offsets[e + 1]) == __ldg(&expert_offsets[e])) {
    return;
  }
  __shared__ __align__(16) uint32_t s_words[64][12];
  __shared__ __align__(16) uint4 s_rem[kFoldRem1 ? 48 : 1];

  const int s_chunk = static_cast<int>(blockIdx.x);
  const int mb = static_cast<int>(blockIdx.y);
  const int nss = k >> 9;
  const size_t w_mb_bytes = static_cast<size_t>(k) * 6;
  const size_t mb_global = static_cast<size_t>(e) * (n >> 4) + mb;
  const uint8_t* dst_mb = prepacked_weights + mb_global * w_mb_bytes;

  const int tid = static_cast<int>(threadIdx.x);
  if constexpr (!kHasRemChunk) {
    if (tid < 192) {
      const int grp = tid >> 5;
      const int slot = PrepackInt3SmSlot(tid & 31, grp & 1);
      *reinterpret_cast<uint4*>(&s_words[slot][(grp >> 1) * 4]) =
          __ldg(reinterpret_cast<const uint4*>(dst_mb + s_chunk * 3072) + tid);
    } else if constexpr (kFoldRem1) {
      if (s_chunk == nss - 1 && tid < 240) {
        const int r_tid = tid - 192;
        s_rem[r_tid] = __ldg(reinterpret_cast<const uint4*>(dst_mb + nss * 3072) + r_tid);
      }
    }
  } else {
    if (s_chunk < nss) {
      if (tid < 192) {
        const int grp = tid >> 5;
        const int slot = PrepackInt3SmSlot(tid & 31, grp & 1);
        *reinterpret_cast<uint4*>(&s_words[slot][(grp >> 1) * 4]) =
            __ldg(reinterpret_cast<const uint4*>(dst_mb + s_chunk * 3072) + tid);
      }
    } else {
      const int rem_steps = (k & 511) >> 7;
      if (tid < rem_steps * 48) {
        reinterpret_cast<uint4*>(s_words)[tid] =
            __ldg(reinterpret_cast<const uint4*>(dst_mb + nss * 3072) + tid);
      }
    }
  }
  __syncthreads();

  const int nh = tid >> 6;           // 0..3
  const int kt = (tid >> 3) & 7;     // 0..7
  const int n_low = (tid >> 1) & 3;  // 0..3
  const int k_half = tid & 1;        // 0..1
  const int k_tiles = k >> 6;
  const int k_tile = (s_chunk << 3) + kt;
  const int num_uint4_per_expert = n * (k >> 5);
  if (!kHasRemChunk || k_tile < k_tiles) {
    const int row_in_mb = (nh << 2) | n_low;  // 0..15
    const int g = row_in_mb >> 1;             // 0..7
    const int odd = row_in_mb & 1;            // 0..1
    const int hk = (kt << 1) | k_half;        // 0..15

    uint32_t w0, w1, w2;
    if (!kHasRemChunk || s_chunk < nss) {
      const int t = (hk >> 2) & 3;
      const int sub = hk & 3;
      const int lane = (g << 2) | t;
      const int slot = PrepackInt3SmSlot(lane, odd);
      w0 = s_words[slot][3 * sub + 0];
      w1 = s_words[slot][3 * sub + 1];
      w2 = s_words[slot][3 * sub + 2];
    } else {
      const int rem_step = hk >> 2;
      const int t = hk & 3;
      const int lane = (g << 2) | t;
      const uint32_t* d_rem = reinterpret_cast<const uint32_t*>(s_words) + rem_step * 192;
      if (odd == 0) {
        w0 = d_rem[lane * 4 + 0];
        w1 = d_rem[lane * 4 + 1];
        w2 = d_rem[lane * 4 + 2];
      } else {
        w0 = d_rem[lane * 4 + 3];
        w1 = d_rem[128 + lane * 2 + 0];
        w2 = d_rem[128 + lane * 2 + 1];
      }
    }

    const int n_high = (mb << 2) + nh;
    const int idx_in_expert = ((n_high * k_tiles + k_tile) << 3) | (n_low << 1) | k_half;
    const size_t out_idx = static_cast<size_t>(e) * num_uint4_per_expert + idx_in_expert;
    reinterpret_cast<uint4*>(sm80_int4_weights)[out_idx] = UnpackInt3WordsToSm80Int4(w0, w1, w2);
  }

  if constexpr (kFoldRem1) {
    if (s_chunk == nss - 1 && tid < 64) {
      const int r_nh = tid >> 4;           // 0..3
      const int r_kt = (tid >> 3) & 1;     // 0..1
      const int r_n_low = (tid >> 1) & 3;  // 0..3
      const int r_k_half = tid & 1;        // 0..1
      const int r_row_in_mb = (r_nh << 2) | r_n_low;
      const int r_g = r_row_in_mb >> 1;
      const int r_odd = r_row_in_mb & 1;
      const int r_t = (r_kt << 1) | r_k_half;
      const int r_lane = (r_g << 2) | r_t;
      const uint32_t* d_rem = reinterpret_cast<const uint32_t*>(s_rem);
      uint32_t rw0, rw1, rw2;
      if (r_odd == 0) {
        rw0 = d_rem[r_lane * 4 + 0];
        rw1 = d_rem[r_lane * 4 + 1];
        rw2 = d_rem[r_lane * 4 + 2];
      } else {
        rw0 = d_rem[r_lane * 4 + 3];
        rw1 = d_rem[128 + r_lane * 2 + 0];
        rw2 = d_rem[128 + r_lane * 2 + 1];
      }
      const int r_k_tile = (nss << 3) + r_kt;
      const int r_n_high = (mb << 2) + r_nh;
      const int r_idx_in_expert = ((r_n_high * k_tiles + r_k_tile) << 3) | (r_n_low << 1) | r_k_half;
      const size_t r_out_idx = static_cast<size_t>(e) * num_uint4_per_expert + r_idx_in_expert;
      reinterpret_cast<uint4*>(sm80_int4_weights)[r_out_idx] = UnpackInt3WordsToSm80Int4(rw0, rw1, rw2);
    }
  }
}

void LaunchRepackRawIntToSm80Int4ActiveExperts(
    const uint8_t* raw_weights, uint8_t* sm80_int4_weights, const int64_t* expert_offsets,
    int num_experts, int n, int k, int src_bits, bool prepacked_int3, cudaStream_t stream) {
  const size_t required_alignment = GetInt2MoePrefillRawWeightAlignment(src_bits);
  ORT_ENFORCE(reinterpret_cast<uintptr_t>(raw_weights) % required_alignment == 0,
              "Raw INT", src_bits, " prefill repack requires ", required_alignment, "-byte aligned weights.");
  if (src_bits == 3 && prepacked_int3) {
    const int nss = k >> 9;
    const int rem_steps = (k & 511) >> 7;
    if (rem_steps == 0) {
      const dim3 grid(static_cast<unsigned>(nss), static_cast<unsigned>(n >> 4), static_cast<unsigned>(num_experts));
      RepackPrepackedInt3ToSm80Int4ActiveExpertsKernel<false, false><<<grid, 256, 0, stream>>>(
          raw_weights, sm80_int4_weights, expert_offsets, n, k);
    } else if (nss > 0 && rem_steps == 1) {
      const dim3 grid(static_cast<unsigned>(nss), static_cast<unsigned>(n >> 4), static_cast<unsigned>(num_experts));
      RepackPrepackedInt3ToSm80Int4ActiveExpertsKernel<true, false><<<grid, 256, 0, stream>>>(
          raw_weights, sm80_int4_weights, expert_offsets, n, k);
    } else {
      const int s_chunks = (k + 511) >> 9;
      const dim3 grid(static_cast<unsigned>(s_chunks), static_cast<unsigned>(n >> 4), static_cast<unsigned>(num_experts));
      RepackPrepackedInt3ToSm80Int4ActiveExpertsKernel<false, true><<<grid, 256, 0, stream>>>(
          raw_weights, sm80_int4_weights, expert_offsets, n, k);
    }
    return;
  }
  const int num_uint4_per_expert = n * (k >> 5);
  const dim3 grid(static_cast<unsigned>((num_uint4_per_expert + 255) / 256), static_cast<unsigned>(num_experts));
  if (src_bits == 2) {
    RepackRawIntToSm80Int4ActiveExpertsKernel<2><<<grid, 256, 0, stream>>>(
        raw_weights, sm80_int4_weights, expert_offsets, n, k);
  } else if (src_bits == 3) {
    RepackRawIntToSm80Int4ActiveExpertsKernel<3><<<grid, 256, 0, stream>>>(
        raw_weights, sm80_int4_weights, expert_offsets, n, k);
  } else {
    ORT_ENFORCE(src_bits == 4, "Raw INT prefill repack requires 2, 3, or 4 bits");
    RepackRawIntToSm80Int4ActiveExpertsKernel<4><<<grid, 256, 0, stream>>>(
        raw_weights, sm80_int4_weights, expert_offsets, n, k);
  }
}

template <typename T, bool kPrepackedInt3 = false>
__global__ void __launch_bounds__(256) TransposeScalesActiveExpertsVec8Kernel(
    const T* __restrict__ raw_scales,
    T* __restrict__ transposed_scales,
    const int64_t* __restrict__ expert_offsets,
    int rows,
    int cols) {
  const int e = static_cast<int>(blockIdx.z);
  if (expert_offsets != nullptr && __ldg(&expert_offsets[e + 1]) == __ldg(&expert_offsets[e])) {
    return;
  }
  __shared__ __align__(16) T s_tile[8][128 + 8];
  const int tile_x = static_cast<int>(blockIdx.x);
  const int tile_y = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);

  const size_t expert_stride = static_cast<size_t>(rows) * cols;
  T* out_e = transposed_scales + static_cast<size_t>(e) * expert_stride;
  const int row_base = tile_y * 128;

  if constexpr (kPrepackedInt3) {
    const int mb_local = tid >> 5;  // 0..7
    const int lane = tid & 31;      // 0..31
    const int nss = cols >> 3;
    const size_t s_mb_bytes = 16ULL * cols * sizeof(T);
    const size_t mb_global = static_cast<size_t>(e) * (rows >> 4) + tile_y * 8 + mb_local;
    const uint8_t* in_mb = reinterpret_cast<const uint8_t*>(raw_scales) + mb_global * s_mb_bytes;
    if (tile_x < nss) {
      const int g = lane >> 2;
      const int t = lane & 3;
      const uint2 v = __ldg(reinterpret_cast<const uint2*>(in_mb + tile_x * 256) + lane);
      const T* sc = reinterpret_cast<const T*>(&v);
      s_tile[2 * t + 0][mb_local * 16 + 2 * g + 0] = sc[0];
      s_tile[2 * t + 1][mb_local * 16 + 2 * g + 0] = sc[1];
      s_tile[2 * t + 0][mb_local * 16 + 2 * g + 1] = sc[2];
      s_tile[2 * t + 1][mb_local * 16 + 2 * g + 1] = sc[3];
    } else {
      const int rem_cols = cols - (nss << 3);
      for (int idx = lane; idx < (rem_cols << 3); idx += 32) {
        const int i = idx >> 4;
        const int g = (idx >> 1) & 7;
        const int h = idx & 1;
        const uint32_t v = __ldg(reinterpret_cast<const uint32_t*>(in_mb + nss * 256) + idx);
        const T* sc = reinterpret_cast<const T*>(&v);
        s_tile[2 * i + h][mb_local * 16 + 2 * g + 0] = sc[0];
        s_tile[2 * i + h][mb_local * 16 + 2 * g + 1] = sc[1];
      }
    }
  } else {
    const int tx = tid & 7;
    const int ty = tid >> 3;
    const T* in_e = raw_scales + static_cast<size_t>(e) * expert_stride;
    const int col_in = tile_x * 8 + tx;
    if (col_in < cols) {
#pragma unroll
      for (int j = 0; j < 128; j += 32) {
        s_tile[tx][ty + j] = __ldg(&in_e[(row_base + ty + j) * cols + col_in]);
      }
    }
  }
  __syncthreads();

  if (tid < 128) {
    const int c = tid >> 4;
    const int r8 = (tid & 15) * 8;
    const int col_out = tile_x * 8 + c;
    if (col_out < cols) {
      *reinterpret_cast<uint4*>(&out_e[col_out * rows + row_base + r8]) =
          *reinterpret_cast<const uint4*>(&s_tile[c][r8]);
    }
  }
}

template <typename T, bool kPrepackedInt3 = false>
__global__ void __launch_bounds__(256) TransposeScalesActiveExpertsKernel(
    const T* __restrict__ raw_scales,
    T* __restrict__ transposed_scales,
    const int64_t* __restrict__ expert_offsets,
    int rows,
    int cols) {
  const int e = static_cast<int>(blockIdx.z);
  if (expert_offsets != nullptr && __ldg(&expert_offsets[e + 1]) == __ldg(&expert_offsets[e])) {
    return;
  }
  __shared__ T s_tile[32][33];
  const int tile_x = static_cast<int>(blockIdx.x);
  const int tile_y = static_cast<int>(blockIdx.y);

  const int tx = static_cast<int>(threadIdx.x) & 31;
  const int ty = static_cast<int>(threadIdx.x) >> 5;

  const size_t expert_stride = static_cast<size_t>(rows) * cols;
  const T* in_e = raw_scales + static_cast<size_t>(e) * expert_stride;
  T* out_e = transposed_scales + static_cast<size_t>(e) * expert_stride;

  const int col_in = tile_x * 32 + tx;
  const int nss = cols >> 3;
  const size_t s_mb_bytes = 16ULL * cols * sizeof(T);
#pragma unroll
  for (int j = 0; j < 32; j += 8) {
    const int row_in = tile_y * 32 + ty + j;
    if (row_in < rows && col_in < cols) {
      if constexpr (kPrepackedInt3) {
        const size_t mb_global = static_cast<size_t>(e) * (rows >> 4) + (row_in >> 4);
        const int g = (row_in >> 1) & 7;
        const int odd = row_in & 1;
        const uint8_t* in_mb = reinterpret_cast<const uint8_t*>(raw_scales) + mb_global * s_mb_bytes;
        if (col_in < (nss << 3)) {
          const int s = col_in >> 3;
          const int t = (col_in >> 1) & 3;
          const int c_low = col_in & 1;
          const int lane = 4 * g + t;
          s_tile[ty + j][tx] = __ldg(&reinterpret_cast<const T*>(in_mb + s * 256)[lane * 4 + odd * 2 + c_low]);
        } else {
          const int i = (col_in - (nss << 3)) >> 1;
          const int h = col_in & 1;
          const int pair = 2 * g + h;
          s_tile[ty + j][tx] = __ldg(&reinterpret_cast<const T*>(in_mb + nss * 256 + i * 64)[pair * 2 + odd]);
        }
      } else {
        s_tile[ty + j][tx] = __ldg(&in_e[row_in * cols + col_in]);
      }
    }
  }
  __syncthreads();

  const int row_out = tile_y * 32 + tx;
#pragma unroll
  for (int j = 0; j < 32; j += 8) {
    const int col_out = tile_x * 32 + ty + j;
    if (col_out < cols && row_out < rows) {
      out_e[col_out * rows + row_out] = s_tile[tx][ty + j];
    }
  }
}

template <typename T>
void LaunchTransposeScalesActiveExperts(
    const T* raw_scales, T* transposed_scales, const int64_t* expert_offsets,
    int num_experts, int rows, int cols, bool prepacked_int3, cudaStream_t stream) {
  if constexpr (sizeof(T) == 2) {
    if ((rows & 127) == 0 && (!prepacked_int3 || (cols & 1) == 0) &&
        (reinterpret_cast<uintptr_t>(transposed_scales) & 15u) == 0) {
      const int tiles_x = (cols + 7) >> 3;
      const int tiles_y = rows >> 7;
      const dim3 grid(static_cast<unsigned>(tiles_x), static_cast<unsigned>(tiles_y), static_cast<unsigned>(num_experts));
      if (prepacked_int3) {
        TransposeScalesActiveExpertsVec8Kernel<T, true><<<grid, 256, 0, stream>>>(
            raw_scales, transposed_scales, expert_offsets, rows, cols);
      } else {
        TransposeScalesActiveExpertsVec8Kernel<T, false><<<grid, 256, 0, stream>>>(
            raw_scales, transposed_scales, expert_offsets, rows, cols);
      }
      return;
    }
  }
  const int tiles_x = (cols + 31) >> 5;
  const int tiles_y = (rows + 31) >> 5;
  const dim3 grid(static_cast<unsigned>(tiles_x), static_cast<unsigned>(tiles_y), static_cast<unsigned>(num_experts));
  if (prepacked_int3) {
    TransposeScalesActiveExpertsKernel<T, true><<<grid, 256, 0, stream>>>(
        raw_scales, transposed_scales, expert_offsets, rows, cols);
  } else {
    TransposeScalesActiveExpertsKernel<T, false><<<grid, 256, 0, stream>>>(
        raw_scales, transposed_scales, expert_offsets, rows, cols);
  }
}

template <typename ElementType>
void RunPackedGroupedGemm(const Int2MoePrefillParams& params,
                          const ElementType* activations, const uint8_t* weights,
                          const void* scales, const ElementType* biases, const int64_t* offsets,
                          ElementType* output, int64_t num_rows, int num_columns, int reduction_size,
                          int weight_bits) {
  if (weight_bits == 2) {
    ORT_ENFORCE(biases == nullptr, "INT2 grouped GEMM does not support epilogue biases");
    Int2GroupedGemmParamsT<ElementType> gemm;
    gemm.activations = activations;
    gemm.packed_weights = weights;
    gemm.block_scales = static_cast<const ElementType*>(scales);
    gemm.expert_row_ends = offsets + 1;
    gemm.output = output;
    gemm.num_rows = num_rows;
    gemm.num_columns = num_columns;
    gemm.reduction_size = reduction_size;
    gemm.block_size = params.block_size;
    gemm.num_experts = params.num_experts;
    gemm.sm = params.sm;
    gemm.multiprocessor_count = params.multiprocessor_count;
    gemm.stream = params.stream;
    RunInt2GroupedGemm(gemm);
    return;
  }
  ORT_ENFORCE(weight_bits == 4, "Packed INT prefill requires INT2 or INT4 weights");
  GroupedGemmInput<ElementType, cutlass::uint4b_t, ElementType, ElementType> gemm{
      activations, offsets + 1, reinterpret_cast<const cutlass::uint4b_t*>(weights), static_cast<const ElementType*>(scales), nullptr, biases, output, nullptr, nullptr, ActivationType::Identity, num_rows, num_columns, reduction_size, params.num_experts, params.block_size, true, false, params.stream, {}, {}};
  gemm.gemm_config = cutlass_extensions::CutlassGemmConfig(
      cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
      cutlass_extensions::SplitKStyle::NO_SPLIT_K, 1, 4);
  MoeGemmRunner<ElementType, cutlass::uint4b_t, ElementType> runner(params.sm, params.multiprocessor_count);
  runner.moeGemm(gemm, {});
}

template <typename ElementType>
void RunInt2MoePrefillImpl(const Int2MoePrefillParams& params, void* workspace) {
  const PrefillWorkspace layout(params);
  auto* storage = static_cast<char*>(workspace);
  auto* permuted_rows = reinterpret_cast<int*>(storage + layout.permuted_rows);
  auto* permuted_experts = reinterpret_cast<int*>(storage + layout.permuted_experts);
  auto* offsets = reinterpret_cast<int64_t*>(storage + layout.expert_offsets);
  auto* expanded_input = reinterpret_cast<ElementType*>(storage + layout.expanded_input);
  auto* fc1_output = reinterpret_cast<ElementType*>(storage + layout.fc1_output);
  auto* activated_output = reinterpret_cast<ElementType*>(storage + layout.activated_output);
  auto* fc2_output = reinterpret_cast<ElementType*>(storage + layout.fc2_output);
  const int64_t expanded = params.num_rows * params.top_k;

  // Raw (INT3) weights fold the expert biases into the GEMM epilogue, which adds them to the FP32 accumulators and
  // rounds once, like the SIMT kernels. The packed INT2/INT4 path keeps adding them after the GEMM result is stored.
  const auto* fc1_bias = static_cast<const ElementType*>(params.fc1_bias);
  const auto* fc2_bias = static_cast<const ElementType*>(params.fc2_bias);
  const ElementType* fc1_gemm_bias = params.weights_are_raw ? fc1_bias : nullptr;
  const ElementType* fc1_activation_bias = params.weights_are_raw ? nullptr : fc1_bias;
  const ElementType* fc2_gemm_bias = params.weights_are_raw ? fc2_bias : nullptr;
  const ElementType* fc2_finalize_bias = params.weights_are_raw ? nullptr : fc2_bias;

  threeStepBuildExpertMapsSortFirstToken(
      params.selected_experts, permuted_experts, permuted_rows, params.unpermuted_to_permuted, offsets,
      reinterpret_cast<int*>(storage + layout.counts), reinterpret_cast<int*>(storage + layout.cumulative_counts),
      reinterpret_cast<int*>(storage + layout.blocked_rows), params.num_rows, params.num_experts, params.top_k,
      0, params.stream);
  const QuantParams quant_params{};
  expandInputRowsKernelLauncher<ElementType, ElementType>(
      static_cast<const ElementType*>(params.input), expanded_input, nullptr, nullptr, permuted_rows,
      params.num_rows, params.hidden_size,
      params.top_k, params.num_experts, quant_params, false, offsets, nullptr, nullptr, nullptr, params.stream);

  if (params.weights_are_raw) {
    auto* repacked_weights = reinterpret_cast<uint8_t*>(storage + layout.repacked_weights);
    auto* transposed_scales = reinterpret_cast<ElementType*>(storage + layout.transposed_scales);
    const int fc1_n = params.inter_size * 2;
    const int fc1_k = params.hidden_size;
    LaunchRepackRawIntToSm80Int4ActiveExperts(
        params.fc1_weights, repacked_weights, offsets, params.num_experts,
        fc1_n, fc1_k, params.fc1_weight_bits, params.weights_are_prepacked_int3, params.stream);
    LaunchTransposeScalesActiveExperts<ElementType>(
        static_cast<const ElementType*>(params.fc1_scales), transposed_scales, offsets,
        params.num_experts, fc1_n, fc1_k / params.block_size, params.weights_are_prepacked_int3, params.stream);
    RunPackedGroupedGemm(params, expanded_input, repacked_weights, transposed_scales, fc1_gemm_bias,
                         offsets, fc1_output, expanded, fc1_n, fc1_k, /*weight_bits=*/4);
  } else {
    RunPackedGroupedGemm(params, expanded_input, params.fc1_weights, params.fc1_scales, fc1_gemm_bias,
                         offsets, fc1_output, expanded, params.inter_size * 2,
                         params.hidden_size, params.fc1_weight_bits);
  }

  ActivationParams activation(ActivationType::Swiglu);
  activation.alpha = params.alpha;
  activation.beta = params.beta;
  activation.limit = params.limit;
  activation.swiglu_fusion = 1;
  doActivation<ElementType, ElementType, ElementType>(
      activated_output, fc1_output, nullptr, fc1_activation_bias, true, offsets,
      params.num_experts, params.inter_size, expanded, ActivationType::Swiglu,
      quant_params, false, nullptr, params.stream, activation);

  if (params.weights_are_raw) {
    auto* repacked_weights = reinterpret_cast<uint8_t*>(storage + layout.repacked_weights);
    auto* transposed_scales = reinterpret_cast<ElementType*>(storage + layout.transposed_scales);
    const int fc2_n = params.hidden_size;
    const int fc2_k = params.inter_size;
    LaunchRepackRawIntToSm80Int4ActiveExperts(
        params.fc2_weights, repacked_weights, offsets, params.num_experts,
        fc2_n, fc2_k, params.fc2_weight_bits, params.weights_are_prepacked_int3, params.stream);
    LaunchTransposeScalesActiveExperts<ElementType>(
        static_cast<const ElementType*>(params.fc2_scales), transposed_scales, offsets,
        params.num_experts, fc2_n, fc2_k / params.block_size, params.weights_are_prepacked_int3, params.stream);
    RunPackedGroupedGemm(params, activated_output, repacked_weights, transposed_scales, fc2_gemm_bias,
                         offsets, fc2_output, expanded, fc2_n, fc2_k, /*weight_bits=*/4);
  } else {
    RunPackedGroupedGemm(params, activated_output, params.fc2_weights, params.fc2_scales, fc2_gemm_bias,
                         offsets, fc2_output, expanded, params.hidden_size,
                         params.inter_size, params.fc2_weight_bits);
  }
  finalizeMoeRoutingKernelLauncher<ElementType, ElementType, ElementType>(
      fc2_output, static_cast<ElementType*>(params.output), fc2_finalize_bias,
      params.routing_weights, params.unpermuted_to_permuted,
      permuted_rows, params.selected_experts, offsets, params.num_rows, params.hidden_size, params.top_k,
      params.num_experts, MOEParallelismConfig{}, false, params.stream);
}

}  // namespace

size_t GetInt2MoePrefillRawWeightAlignment(int weight_bits) {
  // The widest load RepackRawIntToSm80Int4ActiveExpertsKernel issues on the raw weights, per bit width.
  switch (weight_bits) {
    case 2:
      return alignof(uint2);
    case 3:
      return alignof(uint32_t);
    case 4:
      return alignof(uint4);
    default:
      ORT_THROW("Raw INT prefill repack requires 2, 3, or 4 bits, got ", weight_bits);
  }
}

size_t GetInt2MoePrefillWorkspaceSize(const Int2MoePrefillParams& params) {
  return PrefillWorkspace(params).bytes;
}

void RunInt2MoePrefill(const Int2MoePrefillParams& params, void* workspace) {
  if (params.is_bf16) {
#ifdef ENABLE_BF16
    RunInt2MoePrefillImpl<__nv_bfloat16>(params, workspace);
#else
    ORT_THROW("BF16 INT2 prefill requires ENABLE_BF16");
#endif
    return;
  }
  RunInt2MoePrefillImpl<half>(params, workspace);
}

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
