// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tensor-core (mma.sync.m16n8k16, BF16 inputs, FP32 accumulation) INT4 block-wise (group size 32)
// MoE decode GEMV.
//
// The scalar GEMV in moe_gemv.cu launches one block per expanded row, so an expert that serves r
// rows dequantizes (and streams) its weights r times. With S=8 speculative-decode tokens and top-10
// routing there are 80 rows over only ~16-25 distinct experts, which makes that kernel
// instruction-issue bound (about 5 thread instructions per weight per row). This kernel instead
// assigns up to 8 consecutive rows of the same expert (one mma N-tile) to a block: each weight is
// loaded and dequantized exactly once and the per-row dot products run on the tensor cores.
//
// Mapping (per warp, per 128-wide K step):
//   * mma A (16x16) = weights. Row g is the even real column 2*(c0+g) and row g+8 the odd real
//     column of the same pair, so a lane owns the (gate, up) pair of the interleaved SwiGLU layout.
//   * mma B (16x8)  = activations of up to 8 routes of one expert (lane (g, t) loads route g).
//   * Lane (g, t) loads, with one LDG.128 per column slot, the 32-wide k group t of the weight
//     layout (ColumnMajorInterleaved, 64-wide K tiles, 4 columns per interleaved row). Words of that
//     16-byte unit hold the logical elements {2j, 2j+1, 2j+8, 2j+9, 2j+16, ...}; pair ii of word j is
//     logical pair q = 4*ii + j, so the logical<->physical permutation is static register renaming
//     and the matching activation pairs are comp j of the lane's uint4 number ii.
//   * The group scale is applied to the dequantized weights before the MMA (one bf16 rounding per
//     weight, the same contract as the CUTLASS grouped-GEMM path), because the four k groups of one
//     MMA live in different lanes.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace onnxruntime::llm::kernels::fpA_intB_gemv {
namespace moe_tc {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxRoutes = 8;  // mma N
constexpr int kMmaM = 16;      // output columns per m-block
constexpr int kStepK = 128;    // K elements consumed per warp step (4 lanes x one 32-wide group)

struct Params {
  const __nv_bfloat16* act;
  const uint8_t* weight;
  const __nv_bfloat16* scales;
  const __nv_bfloat16* bias;
  __nv_bfloat16* out;
  const int64_t* expert_first_token_offset;
  const int* permuted_row_to_expert;
  const int* permuted_row_to_source_row;
  const float* swiglu_alpha;
  const float* swiglu_beta;
  const float* swiglu_limit;
  float alpha;
  float beta;
  float limit;
  int64_t weight_expert_stride;  // bytes
  int64_t scale_expert_stride;   // elements
  int num_experts;
  int n;  // real output columns (2 * inter_size for the interleaved SwiGLU kernel)
  int k;
  int num_rows;
};

#if defined(__CUDACC__)

__device__ __forceinline__ uint4 LoadWeightsStreaming(const void* ptr) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr));
  return v;
}

template <int J>
__device__ __forceinline__ uint32_t Comp(const uint4& v) {
  if constexpr (J == 0) {
    return v.x;
  } else if constexpr (J == 1) {
    return v.y;
  } else if constexpr (J == 2) {
    return v.z;
  } else {
    return v.w;
  }
}

__device__ __forceinline__ uint32_t Bf16x2Mul(uint32_t a, uint32_t b) {
  uint32_t d;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
#else
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
#endif
  return d;
}

// Nibbles (Shift/4) and (Shift/4 + 4) of `w` become bf16x2 {(c_lo - 8) * s, (c_hi - 8) * s}.
template <int Shift>
__device__ __forceinline__ uint32_t DequantPair(uint32_t w, uint32_t scale2) {
  uint32_t v;
  const uint32_t x = (Shift == 0) ? w : (w >> Shift);
  // (x & 0x000F000F) | 0x43004300 == bf16x2 {128 + c_lo, 128 + c_hi}
  asm("lop3.b32 %0, %1, %2, %3, %4;\n"
      : "=r"(v)
      : "r"(x), "n"(0x000F000F), "n"(0x43004300), "n"((0xf0 & 0xcc) | 0xaa));
  uint32_t r;
  // {-136, -136} in bf16: exact (c - 8).
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(v), "r"(0x3F803F80u), "r"(0xC308C308u));
  return Bf16x2Mul(r, scale2);
}

__device__ __forceinline__ void MmaBf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// One 32-bit weight word of the even (row g) and odd (row g+8) column feeds two MMA steps; word j
// holds logical pairs q = j, 4+j, 8+j, 12+j, matched with comp j of activation uint4 0..3.
template <int J, int kM>
__device__ __forceinline__ void ComputeWord(float (&acc)[kM][4], int mb, const uint4& we, const uint4& wo,
                                            uint32_t se, uint32_t so, const uint4 (&b)[4]) {
  const uint32_t ew = Comp<J>(we);
  const uint32_t ow = Comp<J>(wo);
  const uint32_t e0 = DequantPair<0>(ew, se);
  const uint32_t o0 = DequantPair<0>(ow, so);
  const uint32_t e1 = DequantPair<4>(ew, se);
  const uint32_t o1 = DequantPair<4>(ow, so);
  MmaBf16(acc[mb], e0, o0, e1, o1, Comp<J>(b[0]), Comp<J>(b[1]));
  const uint32_t e2 = DequantPair<8>(ew, se);
  const uint32_t o2 = DequantPair<8>(ow, so);
  const uint32_t e3 = DequantPair<12>(ew, se);
  const uint32_t o3 = DequantPair<12>(ow, so);
  MmaBf16(acc[mb], e2, o2, e3, o3, Comp<J>(b[2]), Comp<J>(b[3]));
}

// kSwiglu = true : FC1 with interleaved (gate, up) columns; output [row][n / 2] after SwiGLU.
// kSwiglu = false: plain GEMV; output [row][n].
// kWarpsK warps split K (reduced through shared memory), kWarps / kWarpsK warps split the columns,
// and every warp owns kM m-blocks (16 columns each).
template <bool kSwiglu, int kWarpsK, int kM>
__global__ void __launch_bounds__(kThreads) MoeGemvTcInt4Kernel(const Params p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  static_assert(kWarps % kWarpsK == 0, "warps must split evenly over K");
  constexpr int kWarpsN = kWarps / kWarpsK;
  constexpr int kTileN = kWarpsN * kM * kMmaM;

  const int row0 = blockIdx.x;
  const int expert = p.permuted_row_to_expert[row0];
  if (expert < 0 || expert >= p.num_experts) {
    return;
  }
  const int64_t first = p.expert_first_token_offset[expert];
  const int64_t last = p.expert_first_token_offset[expert + 1];
  // Rows of one expert are contiguous; every kMaxRoutes-th row of a run leads a chunk.
  if (((row0 - first) & (kMaxRoutes - 1)) != 0) {
    return;
  }
  const int routes = (last - row0) < kMaxRoutes ? static_cast<int>(last - row0) : kMaxRoutes;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int wk = warp % kWarpsK;
  const int wn = warp / kWarpsK;
  const int n = p.n;
  const int k = p.k;
  const int col_w = blockIdx.y * kTileN + wn * (kM * kMmaM);
  const int steps = (k / kStepK) / kWarpsK;
  const int step0 = wk * steps;

  // An interleaved weight row holds 4 real columns (k * 4 bits each = 2k bytes); inside a 64-wide K
  // tile the columns are 32-byte segments, and the two 16-byte halves of a segment are k groups.
  const int64_t row_bytes = 2 * static_cast<int64_t>(k);
  const uint8_t* wptr = p.weight + static_cast<int64_t>(expert) * p.weight_expert_stride +
                        (static_cast<int64_t>(col_w / 4) + (g >> 1)) * row_bytes + (g & 1) * 64 +
                        static_cast<int64_t>(step0) * 256 + (t >> 1) * 128 + (t & 1) * 16;
  const __nv_bfloat16* sptr = p.scales + static_cast<int64_t>(expert) * p.scale_expert_stride +
                              (static_cast<int64_t>(step0) * 4 + t) * n + col_w + 2 * g;
  const __nv_bfloat16* aptr = nullptr;
  if (g < routes) {
    int src_row = row0 + g;
    if constexpr (kSwiglu) {
      if (p.permuted_row_to_source_row != nullptr) {
        src_row = p.permuted_row_to_source_row[row0 + g] % p.num_rows;
      }
    }
    aptr = p.act + static_cast<int64_t>(src_row) * k + static_cast<int64_t>(step0) * kStepK + 32 * t;
  }

  float acc[kM][4];
#pragma unroll
  for (int mb = 0; mb < kM; ++mb) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc[mb][i] = 0.f;
    }
  }
  uint4 b[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    b[i] = make_uint4(0u, 0u, 0u, 0u);
  }

#pragma unroll 1
  for (int s = 0; s < steps; ++s) {
    uint4 we[kM], wo[kM];
    uint32_t sc[kM];
#pragma unroll
    for (int mb = 0; mb < kM; ++mb) {
      const uint8_t* w = wptr + static_cast<int64_t>(mb) * 4 * row_bytes;
      we[mb] = LoadWeightsStreaming(w);
      wo[mb] = LoadWeightsStreaming(w + 32);
      sc[mb] = *reinterpret_cast<const uint32_t*>(sptr + mb * kMmaM);
    }
    if (g < routes) {
      const uint4* a4 = reinterpret_cast<const uint4*>(aptr);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        b[i] = a4[i];
      }
    }
#pragma unroll
    for (int mb = 0; mb < kM; ++mb) {
      const uint32_t se = __byte_perm(sc[mb], 0, 0x1010);
      const uint32_t so = __byte_perm(sc[mb], 0, 0x3232);
      ComputeWord<0, kM>(acc, mb, we[mb], wo[mb], se, so, b);
      ComputeWord<1, kM>(acc, mb, we[mb], wo[mb], se, so, b);
      ComputeWord<2, kM>(acc, mb, we[mb], wo[mb], se, so, b);
      ComputeWord<3, kM>(acc, mb, we[mb], wo[mb], se, so, b);
    }
    wptr += 256;
    sptr += 4 * static_cast<int64_t>(n);
    aptr += kStepK;
  }

  const __nv_bfloat16* bias = p.bias != nullptr ? p.bias + static_cast<int64_t>(expert) * n : nullptr;
  float alpha = p.alpha;
  float beta = p.beta;
  float limit = p.limit;
  if constexpr (kSwiglu) {
    alpha = p.swiglu_alpha != nullptr ? p.swiglu_alpha[expert] : alpha;
    beta = p.swiglu_beta != nullptr ? p.swiglu_beta[expert] : beta;
    limit = p.swiglu_limit != nullptr ? p.swiglu_limit[expert] : limit;
  }

  // d = {(even col, route 2t), (even col, route 2t+1), (odd col, route 2t), (odd col, route 2t+1)}
  auto finalize = [&](const float4 d, int wn2, int mb, int l) {
    const int g2 = l >> 2;
    const int t2 = l & 3;
    const int col = blockIdx.y * kTileN + (wn2 * kM + mb) * kMmaM + 2 * g2;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int route = 2 * t2 + h;
      if (route >= routes) {
        continue;
      }
      float even = h == 0 ? d.x : d.y;
      float odd = h == 0 ? d.z : d.w;
      if (bias != nullptr) {
        even += __bfloat162float(bias[col]);
        odd += __bfloat162float(bias[col + 1]);
      }
      if constexpr (kSwiglu) {
        float gate = even;
        float linear = odd;
        if (isfinite(limit)) {
          gate = fminf(gate, limit);
          linear = fminf(fmaxf(linear, -limit), limit);
        }
        linear += beta;
        const float sigmoid = 1.0f / (1.0f + expf(-alpha * gate));
        p.out[static_cast<int64_t>(row0 + route) * (n / 2) + col / 2] =
            __float2bfloat16_rn(gate * sigmoid * linear);
      } else {
        *reinterpret_cast<__nv_bfloat162*>(p.out + static_cast<int64_t>(row0 + route) * n + col) =
            __floats2bfloat162_rn(even, odd);
      }
    }
  };

  if constexpr (kWarpsK > 1) {
    __shared__ float4 red[kWarpsK][kWarpsN][kM][32];
#pragma unroll
    for (int mb = 0; mb < kM; ++mb) {
      red[wk][wn][mb][lane] = make_float4(acc[mb][0], acc[mb][1], acc[mb][2], acc[mb][3]);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < kWarpsN * kM * 32; idx += kThreads) {
      const int l = idx & 31;
      const int mb = (idx >> 5) % kM;
      const int wn2 = (idx >> 5) / kM;
      float4 d = red[0][wn2][mb][l];
#pragma unroll
      for (int w = 1; w < kWarpsK; ++w) {
        const float4 e = red[w][wn2][mb][l];
        d.x += e.x;
        d.y += e.y;
        d.z += e.z;
        d.w += e.w;
      }
      finalize(d, wn2, mb, l);
    }
  } else {
#pragma unroll
    for (int mb = 0; mb < kM; ++mb) {
      finalize(make_float4(acc[mb][0], acc[mb][1], acc[mb][2], acc[mb][3]), wn, mb, lane);
    }
  }
#endif
}

// The kernel body is compiled for SM80+ only (BF16 mma.sync), so older devices must keep the scalar GEMV.
inline bool DeviceSupportsBf16Mma() {
  int device = 0;
  int major = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
    return false;
  }
  return major >= 8;
}

template <bool kSwiglu, int kWarpsK, int kM>
inline void Launch(const Params& p, int64_t expanded_num_rows, cudaStream_t stream) {
  constexpr int kTileN = (kWarps / kWarpsK) * kM * kMmaM;
  const dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(p.n / kTileN));
  MoeGemvTcInt4Kernel<kSwiglu, kWarpsK, kM><<<grid, kThreads, 0, stream>>>(p);
}

// Returns true when the tensor-core kernel handled the launch; false leaves the work to the scalar
// kernel. `n` is the number of real output columns (2 * inter_size for the interleaved SwiGLU FC1).
inline bool TryLaunchInt4Bf16(bool swiglu, const __nv_bfloat16* act, const uint8_t* weight,
                              const __nv_bfloat16* scales, const __nv_bfloat16* bias, __nv_bfloat16* out,
                              const int64_t* expert_first_token_offset, const int* permuted_row_to_expert,
                              int num_experts, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                              const float* swiglu_alpha, const float* swiglu_beta, const float* swiglu_limit,
                              float alpha, float beta, float limit, const int* permuted_row_to_source_row,
                              int64_t num_rows, cudaStream_t stream, int tensor_core_min_rows) {
  if (group_size != 32 || permuted_row_to_expert == nullptr || expert_first_token_offset == nullptr ||
      expanded_num_rows < tensor_core_min_rows || tensor_core_min_rows == 0 || !DeviceSupportsBf16Mma()) {
    return false;
  }
  const bool shape_ok = swiglu ? (n % 64 == 0 && k % 512 == 0) : (n % 128 == 0 && k % 128 == 0);
  if (!shape_ok) {
    return false;
  }

  cudaFuncAttributes attributes{};
  const auto kernel = swiglu ? MoeGemvTcInt4Kernel<true, 4, 4> : MoeGemvTcInt4Kernel<false, 1, 2>;
  if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess || attributes.ptxVersion < 80) {
    return false;
  }

  Params p;
  p.act = act;
  p.weight = weight;
  p.scales = scales;
  p.bias = bias;
  p.out = out;
  p.expert_first_token_offset = expert_first_token_offset;
  p.permuted_row_to_expert = permuted_row_to_expert;
  p.permuted_row_to_source_row = permuted_row_to_source_row;
  p.swiglu_alpha = swiglu_alpha;
  p.swiglu_beta = swiglu_beta;
  p.swiglu_limit = swiglu_limit;
  p.alpha = alpha;
  p.beta = beta;
  p.limit = limit;
  p.weight_expert_stride = n * k / 2;
  p.scale_expert_stride = (k / group_size) * n;
  p.num_experts = num_experts;
  p.n = static_cast<int>(n);
  p.k = static_cast<int>(k);
  p.num_rows = static_cast<int>(num_rows);

  if (swiglu) {
    Launch<true, 4, 4>(p, expanded_num_rows, stream);
  } else {
    Launch<false, 1, 2>(p, expanded_num_rows, stream);
  }
  return true;
}

#endif  // __CUDACC__

}  // namespace moe_tc
}  // namespace onnxruntime::llm::kernels::fpA_intB_gemv
