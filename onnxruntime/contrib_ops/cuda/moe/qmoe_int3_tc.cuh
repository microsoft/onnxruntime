// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tensor-core (mma.sync.m16n8k16, BF16 inputs, FP32 accumulation) INT3 block-wise (block size 64) QMoE decode.
//
// The vectorized decode GEMV (FusedFc1FastVecKernel / Fc2RouteFastVecKernel) shares dequantized weights between at
// most two routes, so with speculative decoding (e.g. 8 tokens x top-10 = 80 routes over ~16-40 experts) every
// expert is dequantized several times and the kernels are instruction-issue bound. This kernel gives one CTA up to
// eight routes of the same expert (one mma N-tile): each weight is loaded and dequantized once and the dot products
// run on the tensor cores.
//
// Mapping (m16n8k16, per warp):
//   * mma A (16x16) = weights. A row g is the even real row (gate for FC1) and row g + 8 the next odd real row (up),
//     i.e. the two rows of one interleaved SwiGLU pair are adjacent in memory.
//   * mma B (16x8)  = activations of up to eight routes of one expert (lane (g, t) serves route slot g).
//   * The row-local 3-bit bitstream (code i at bit 3 i) makes 128 codes = 48 bytes the smallest 16-byte aligned unit,
//     so lane (g, t) of a quad owns one such 48-byte chunk per 512-wide K super-step. K positions inside an mma are
//     a free permutation as long as A and B agree: the pair (code j, code j + 4) of every group of 8 codes is one
//     packed bf16x2 A register (exactly the pairing of DequantizeInt3x8Bf16), and the activations are pre-permuted
//     in shared memory to {x0,x4}, {x1,x5}, {x2,x6}, {x3,x7} so a group's four B registers are one LDS.128.
//   * The dequantization rounds once to bf16 per weight like the GEMV kernels (one fused multiply-add on the
//     subnormal-encoded codes, see kDequantShift), so weights are bit-identical to the GEMV path up to an exact power
//     of two that finalize() removes; only the fp32 accumulation order differs.
// Route groups are derived in-kernel from the per-route expert ids: every run of up to 8 routes (in route order) of
// one expert is led by its first route; CTAs of non-leading routes exit immediately.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "contrib_ops/cuda/moe/qmoe_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace int3_tc {

constexpr int kMaxRoutes = 8;         // mma N
constexpr int kMaxTotalRoutes = 256;  // route ids are packed into bytes
constexpr int kBlockSize = 64;        // quantization block (codes per scale)
constexpr int kStepK = 512;           // K consumed per super-step: 4 lanes x 128 codes
constexpr int kKPartitions = 5;       // super-steps are summed in this many partial sums (s % kKPartitions)
constexpr int kMaxSharedBytes = 48 * 1024;

struct Params {
  const void* input;            // FC1: [rows][k] (row = route / top_k); FC2: [routes][k]
  void* out;                    // FC1: [routes][n / 2]; FC2: [routes][n]
  const uint8_t* weights;       // [experts][n][k * 3 / 8]
  const __nv_bfloat16* scales;  // [experts * n][k / 64]
  const __nv_bfloat16* bias;    // [experts][n] or nullptr
  const int* experts;           // [total_routes]
  const int* source_route_ids;  // optional [total_routes] source row map for FC1
  const int* sorted_routes;     // optional [total_routes] grouped route ids
  const Int3ExpertTileDesc* tile_descs;
  const int* num_tiles_ptr;
  bool input_is_gathered;
  int total_routes;
  int num_experts;
  int top_k;
  int n;  // real output rows (2 * intermediate for FC1)
  int k;
  float alpha;
  float beta;
  float limit;
};

#if defined(__CUDACC__)

__device__ __forceinline__ uint4 LoadWeights128(const void* ptr) {
  uint4 v;
  asm("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint32_t LoadWord(const void* ptr) {
  uint32_t v;
  asm("ld.global.nc.u32 %0, [%1];\n" : "=r"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint32_t LoadHalfWord(const void* ptr) {
  unsigned short v;
  asm("ld.global.nc.u16 %0, [%1];\n" : "=h"(v) : "l"(ptr));
  return v;
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

// 32 bits of the 96-bit unit (three words = 32 codes) starting at bit S. Code j of a group of eight lands at window
// bit 4 and code j + 4 at bit 16, which is the layout DequantizeInt3x8Bf16 feeds to its LOP3.
template <int S>
__device__ __forceinline__ uint32_t UnitWindow(const uint32_t* w) {
  if constexpr (S < 0) {
    return w[0] << (-S);
  } else if constexpr ((S & 31) + 19 <= 32) {
    return w[S >> 5] >> (S & 31);
  } else {
    return __funnelshift_r(w[S >> 5], w[(S >> 5) + 1], S & 31);
  }
}

// The 3-bit codes are decoded as subnormal bf16 values (exponent field 0, the code in the mantissa) so that one fused
// multiply-add yields (code - 4) * scale * 2^-kDequantShift with a single rounding. That value is the old exact
// (code - 4) * scale rounded once, scaled by a power of two, so the tensor core results are the old ones times
// 2^-kDequantShift and finalize() undoes it exactly. The multiplier scale * 2^(133 - kDequantShift) must be finite
// (scale < 2^55) and the smallest weight scale * 2^-kDequantShift must be a normal bf16 number (scale >= 2^-66). The
// shift of 60 places that window around the scales real models use; zero is exact. A model with scales outside it
// needs ep.cuda.disable_qmoe_int3_tc=1.
constexpr int kDequantShift = 60;
constexpr float kDequantRescale = static_cast<float>(1ull << kDequantShift);

// bf16 bit pattern of 2^e (e + 127 in the exponent field).
__host__ __device__ constexpr uint32_t Bf16Pow2Bits(int e) { return static_cast<uint32_t>(127 + e) << 7; }

// {fma-multiplier, fma-addend} of a bf16x2 whose two halves hold the same scale.
__device__ __forceinline__ void DequantOperands(uint32_t scale2, uint32_t& mul, uint32_t& add) {
  // The code at mantissa bit 4 (low half) is code * 2^-129 and the one at bit 0 (high half) code * 2^-133.
  constexpr uint32_t kScaleMulBits =
      (Bf16Pow2Bits(133 - kDequantShift) << 16) | Bf16Pow2Bits(129 - kDequantShift);
  // -4 * 2^-kDequantShift in both halves (the sign bit is 0x8000).
  constexpr uint32_t kScaleAddBits = (0x8000u | Bf16Pow2Bits(2 - kDequantShift)) * 0x10001u;
  const __nv_bfloat162 s = *reinterpret_cast<const __nv_bfloat162*>(&scale2);
  const __nv_bfloat162 m = __hmul2(s, *reinterpret_cast<const __nv_bfloat162*>(&kScaleMulBits));
  const __nv_bfloat162 a = __hmul2(s, *reinterpret_cast<const __nv_bfloat162*>(&kScaleAddBits));
  mul = *reinterpret_cast<const uint32_t*>(&m);
  add = *reinterpret_cast<const uint32_t*>(&a);
}

// bf16x2 {bf16((c_lo - 4) * s * 2^-kDequantShift), bf16((c_hi - 4) * s * 2^-kDequantShift)} for the 3-bit codes at
// window bits 4..6 and 16..18.
__device__ __forceinline__ uint32_t DequantPair(uint32_t window, uint32_t mul, uint32_t add) {
  uint32_t bits;
  asm("and.b32 %0, %1, %2;" : "=r"(bits) : "r"(window), "n"(0x00070070u));
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(bits), "r"(mul), "r"(add));
  return r;
}

// One group of eight codes (24 bits at unit bit 24 A) of the even/odd row against one 16-byte activation piece.
template <int A>
__device__ __forceinline__ void GroupMma(float (&acc)[4], const uint32_t* we, const uint32_t* wo, uint32_t se_mul,
                                         uint32_t se_add, uint32_t so_mul, uint32_t so_add, const uint4& b) {
  constexpr int S = 24 * A;
  const uint32_t e0 = DequantPair(UnitWindow<S - 4>(we), se_mul, se_add);
  const uint32_t o0 = DequantPair(UnitWindow<S - 4>(wo), so_mul, so_add);
  const uint32_t e1 = DequantPair(UnitWindow<S - 1>(we), se_mul, se_add);
  const uint32_t o1 = DequantPair(UnitWindow<S - 1>(wo), so_mul, so_add);
  MmaBf16(acc, e0, o0, e1, o1, b.x, b.y);
  const uint32_t e2 = DequantPair(UnitWindow<S + 2>(we), se_mul, se_add);
  const uint32_t o2 = DequantPair(UnitWindow<S + 2>(wo), so_mul, so_add);
  const uint32_t e3 = DequantPair(UnitWindow<S + 5>(we), se_mul, se_add);
  const uint32_t o3 = DequantPair(UnitWindow<S + 5>(wo), so_mul, so_add);
  MmaBf16(acc, e2, o2, e3, o3, b.z, b.w);
}

// kSwiglu = true : FC1, interleaved (gate, up) rows, output [route][n / 2] after SwiGLU.
// kSwiglu = false: FC2, output [route][n].
// Every warp owns kM m-blocks (16 real rows each) and runs over the whole K, so the warps of a CTA are independent.
// FC1 walks its (m-block, super-step) items one after the other with two register sets of weights: the next item is
// requested before the current one is computed, which hides the weight latency behind the dequantization and halves
// the register footprint. The partial sums are kept per K partition (the super-steps s % kKPartitions, as in the
// K-split kernel this schedule replaces) and added in partition order, so the results are bit-identical to it.
// FC2 (one super-step plus a remainder) requests the weights of all its m-blocks up front.
template <bool kSwiglu, int kWarps, int kM, int kMinCtas = 2>
__global__ void __launch_bounds__(kWarps * 32, kMinCtas) Int3TcKernel(const Params p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  static_assert(!kSwiglu || kM % 2 == 0, "the FC1 schedule handles items in pairs");
  constexpr int kThreads = kWarps * 32;
  constexpr int kTileN = kWarps * kM * 16;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;

  int expert = -1;
  int count = 0;
  uint64_t members = 0;
  int grouped_routes[kMaxRoutes];
  if (p.tile_descs != nullptr) {
    const int tile_idx = blockIdx.x;
    if (tile_idx >= __ldg(p.num_tiles_ptr)) {
      return;
    }
    const Int3ExpertTileDesc desc = p.tile_descs[tile_idx];
    expert = desc.expert;
    count = desc.count;
    if (expert < 0 || expert >= p.num_experts || count <= 0) {
      return;
    }
#pragma unroll
    for (int i = 0; i < kMaxRoutes; ++i) {
      grouped_routes[i] = (i < count) ? __ldg(&p.sorted_routes[desc.start_idx + i]) : 0;
    }
  } else {
    // Routes of this expert: this CTA leads the run of (up to) eight routes starting at its own route when it is the
    // 8m-th route of the expert. Every warp derives the same result, so exits are CTA-uniform.
    const int r0 = blockIdx.x;
    const int total = p.total_routes;
    expert = __ldg(p.experts + r0);
    int ev[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int idx = i * 32 + lane;
      ev[i] = idx < total ? __ldg(p.experts + idx) : -1;
    }
    if (expert < 0 || expert >= p.num_experts) {
      return;
    }
    const int c0 = r0 >> 5;
    const int b0 = r0 & 31;
    uint32_t match[8];
    int rank = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      match[i] = __ballot_sync(0xffffffffu, ev[i] == expert);
      const uint32_t below = i < c0 ? 0xffffffffu : (i == c0 ? ((1u << b0) - 1u) : 0u);
      rank += __popc(match[i] & below);
    }
    if ((rank & (kMaxRoutes - 1)) != 0) {
      return;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint32_t bits = match[i] & (i < c0 ? 0u : (i == c0 ? (0xffffffffu << b0) : 0xffffffffu));
      while (bits != 0u && count < kMaxRoutes) {
        const int pos = (i << 5) + (__ffs(bits) - 1);
        bits &= bits - 1u;
        members |= static_cast<uint64_t>(pos) << (8 * count);
        ++count;
      }
    }
  }

  const int n = p.n;
  const int k = p.k;
  const int nss = k / kStepK;
  const int rem = (k / 128) & 3;  // 128-code chunks past the last super-step, split over the quad as 32-code units
  const int64_t row_bytes = static_cast<int64_t>(k) * 3 / 8;
  const int nb = k / kBlockSize;
  const int tile_row0 = blockIdx.y * kTileN;
  const int row_base = tile_row0 + warp * (kM * 16);
  const int64_t wrow = static_cast<int64_t>(expert) * n + row_base + 2 * g;
  const uint8_t* wptr = p.weights + wrow * row_bytes + 48 * t;
  const __nv_bfloat16* sptr = p.scales + wrow * nb + 2 * t;

  extern __shared__ uint4 smem_u4[];
  uint8_t* smem = reinterpret_cast<uint8_t*>(smem_u4);

  // The first weights are requested before the activations are staged so both latencies overlap.
  constexpr int kSets = kSwiglu ? 2 : kM;
  uint32_t we[kSets][12];
  uint32_t wo[kSets][12];
  uint32_t se[kSets];
  uint32_t so[kSets];
  auto load_mb = [&](int set, int mb, int s) {
    const uint8_t* pe = wptr + static_cast<int64_t>(mb) * 16 * row_bytes + s * 192;
    const uint8_t* po = pe + row_bytes;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const uint4 ve = LoadWeights128(pe + 16 * i);
      const uint4 vo = LoadWeights128(po + 16 * i);
      we[set][4 * i + 0] = ve.x;
      we[set][4 * i + 1] = ve.y;
      we[set][4 * i + 2] = ve.z;
      we[set][4 * i + 3] = ve.w;
      wo[set][4 * i + 0] = vo.x;
      wo[set][4 * i + 1] = vo.y;
      wo[set][4 * i + 2] = vo.z;
      wo[set][4 * i + 3] = vo.w;
    }
    se[set] = LoadWord(sptr + static_cast<int64_t>(mb) * 16 * nb + 8 * s);
    so[set] = LoadWord(sptr + static_cast<int64_t>(mb) * 16 * nb + nb + 8 * s);
  };
  if (nss > 0) {
    if constexpr (kSwiglu) {
      load_mb(0, 0, 0);
    } else {
#pragma unroll
      for (int mb = 0; mb < kM; ++mb) {
        load_mb(mb, mb, 0);
      }
    }
  }

  // Stage the activations of the group's routes (zero for unused slots) as 16-byte pieces of eight consecutive K
  // positions, pre-permuted to {x0,x4}, {x1,x5}, {x2,x6}, {x3,x7}. Piece u of route slot r lives at
  // (u * 8 + ((r + 2 h(u)) & 7)) * 16 bytes, where h(u) is the quad lane that consumes piece u, so that the readers
  // of one LDS.128 phase hit eight different 16-byte bank groups.
  {
    const int rs = threadIdx.x & 7;
    const __nv_bfloat16* src = nullptr;
    if (rs < count) {
      const int route = (p.tile_descs != nullptr)
                            ? grouped_routes[rs]
                            : static_cast<int>((members >> (8 * rs)) & 0xffu);
      const int row = kSwiglu
                          ? (p.input_is_gathered
                                 ? route
                                 : (p.source_route_ids != nullptr ? __ldg(&p.source_route_ids[route])
                                                                  : (route / p.top_k)))
                          : route;
      src = static_cast<const __nv_bfloat16*>(p.input) + static_cast<int64_t>(row) * k;
    }
    const int pieces = k >> 3;
    const int ss_pieces = nss * 64;
#pragma unroll 4
    for (int u = threadIdx.x >> 3; u < pieces; u += kThreads / 8) {
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (src != nullptr) {
        const uint4 x = *reinterpret_cast<const uint4*>(src + 8 * u);
        v.x = __byte_perm(x.x, x.z, 0x5410);
        v.y = __byte_perm(x.x, x.z, 0x7632);
        v.z = __byte_perm(x.y, x.w, 0x5410);
        v.w = __byte_perm(x.y, x.w, 0x7632);
      }
      const int h = u < ss_pieces ? ((u >> 4) & 3) : ((u >> 2) & 3);
      smem_u4[u * 8 + ((rs + 2 * h) & 7)] = v;
    }
  }
  __syncthreads();

  const uint8_t* b_lane = smem + 2048 * t + ((g + 2 * t) & 7) * 16;
  const __nv_bfloat16* bias = p.bias != nullptr ? p.bias + static_cast<int64_t>(expert) * n : nullptr;

  // d = {(even row, route 2t'), (even row, route 2t' + 1), (odd row, route 2t'), (odd row, route 2t' + 1)}; mb is the
  // m-block index inside the CTA tile.
  auto finalize = [&](const float4 d, int mb, int l) {
    const int g2 = l >> 2;
    const int t2 = l & 3;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int slot = 2 * t2 + h;
      if (slot >= count) {
        continue;
      }
      const int route = (p.tile_descs != nullptr)
                            ? grouped_routes[slot]
                            : static_cast<int>((members >> (8 * slot)) & 0xffu);
      float even = (h == 0 ? d.x : d.y) * kDequantRescale;
      float odd = (h == 0 ? d.z : d.w) * kDequantRescale;
      if constexpr (kSwiglu) {
        const int col = (tile_row0 >> 1) + 8 * mb + g2;
        if (bias != nullptr) {
          even += __bfloat162float(bias[2 * col]);
          odd += __bfloat162float(bias[2 * col + 1]);
        }
        const float gate = fminf(even, p.limit);
        const float up = fminf(fmaxf(odd, -p.limit), p.limit);
        static_cast<__nv_bfloat16*>(p.out)[static_cast<int64_t>(route) * (n >> 1) + col] =
            __float2bfloat16_rn((gate / (1.0f + expf(-p.alpha * gate))) * (up + p.beta));
      } else {
        const int col = tile_row0 + 16 * mb + 2 * g2;
        if (bias != nullptr) {
          even += __bfloat162float(bias[col]);
          odd += __bfloat162float(bias[col + 1]);
        }
        *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(p.out) + static_cast<int64_t>(route) * n + col) =
            __floats2bfloat162_rn(even, odd);
      }
    }
  };

  if constexpr (kSwiglu) {
    // One item = one m-block over one super-step: 4 units of 32 codes per lane.
    auto compute_item = [&](int set, int s, float (&a)[4]) {
      const uint8_t* b_ss = b_lane + 8192 * s;
#pragma unroll
      for (int unit = 0; unit < 4; ++unit) {
        uint4 bp[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          bp[q] = *reinterpret_cast<const uint4*>(b_ss + (unit * 4 + q) * 128);
        }
        uint32_t se_mul, se_add, so_mul, so_add;
        DequantOperands(__byte_perm(se[set], 0, unit < 2 ? 0x1010 : 0x3232), se_mul, se_add);
        DequantOperands(__byte_perm(so[set], 0, unit < 2 ? 0x1010 : 0x3232), so_mul, so_add);
        GroupMma<0>(a, &we[set][3 * unit], &wo[set][3 * unit], se_mul, se_add, so_mul, so_add, bp[0]);
        GroupMma<1>(a, &we[set][3 * unit], &wo[set][3 * unit], se_mul, se_add, so_mul, so_add, bp[1]);
        GroupMma<2>(a, &we[set][3 * unit], &wo[set][3 * unit], se_mul, se_add, so_mul, so_add, bp[2]);
        GroupMma<3>(a, &we[set][3 * unit], &wo[set][3 * unit], se_mul, se_add, so_mul, so_add, bp[3]);
      }
    };

    // Items run in (m-block, K partition, super-step) order; the partition of super-step s is s % kKPartitions.
    const int partitions = nss < kKPartitions ? nss : kKPartitions;
    auto advance = [&](int& mb, int& part, int& s) {
      s += kKPartitions;
      if (s >= nss) {
        if (++part == partitions) {
          part = 0;
          ++mb;
        }
        s = part;
      }
    };
    float a[4] = {0.f, 0.f, 0.f, 0.f};  // chain of the current partition
    float d[4] = {0.f, 0.f, 0.f, 0.f};  // partitions of the current m-block added so far
    // A partition closes with its last super-step and the m-block with its last partition.
    auto finish_item = [&](int mb, int part, int s) {
      if (s + kKPartitions < nss) {
        return;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        d[i] = part == 0 ? a[i] : d[i] + a[i];
        a[i] = 0.f;
      }
      if (part == partitions - 1) {
        finalize(make_float4(d[0], d[1], d[2], d[3]), warp * kM + mb, lane);
      }
    };
    const int items = kM * nss;  // even
    int mb0 = 0;
    int part0 = 0;
    int s0 = 0;
    for (int it = 0; it < items; it += 2) {
      int mb1 = mb0;
      int part1 = part0;
      int s1 = s0;
      advance(mb1, part1, s1);
      int mb2 = mb1;
      int part2 = part1;
      int s2 = s1;
      advance(mb2, part2, s2);
      load_mb(1, mb1, s1);
      compute_item(0, s0, a);
      finish_item(mb0, part0, s0);
      if (it + 2 < items) {
        load_mb(0, mb2, s2);
      }
      compute_item(1, s1, a);
      finish_item(mb1, part1, s1);
      mb0 = mb2;
      part0 = part2;
      s0 = s2;
    }
  } else {
    float acc[kM][4];
#pragma unroll
    for (int mb = 0; mb < kM; ++mb) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc[mb][i] = 0.f;
      }
    }
    for (int ss = 0; ss < nss; ++ss) {
      const uint8_t* b_ss = b_lane + 8192 * ss;
#pragma unroll
      for (int unit = 0; unit < 4; ++unit) {
        uint4 bp[4];
#pragma unroll
        for (int a = 0; a < 4; ++a) {
          bp[a] = *reinterpret_cast<const uint4*>(b_ss + (unit * 4 + a) * 128);
        }
#pragma unroll
        for (int mb = 0; mb < kM; ++mb) {
          uint32_t se_mul, se_add, so_mul, so_add;
          DequantOperands(__byte_perm(se[mb], 0, unit < 2 ? 0x1010 : 0x3232), se_mul, se_add);
          DequantOperands(__byte_perm(so[mb], 0, unit < 2 ? 0x1010 : 0x3232), so_mul, so_add);
          GroupMma<0>(acc[mb], &we[mb][3 * unit], &wo[mb][3 * unit], se_mul, se_add, so_mul, so_add, bp[0]);
          GroupMma<1>(acc[mb], &we[mb][3 * unit], &wo[mb][3 * unit], se_mul, se_add, so_mul, so_add, bp[1]);
          GroupMma<2>(acc[mb], &we[mb][3 * unit], &wo[mb][3 * unit], se_mul, se_add, so_mul, so_add, bp[2]);
          GroupMma<3>(acc[mb], &we[mb][3 * unit], &wo[mb][3 * unit], se_mul, se_add, so_mul, so_add, bp[3]);
        }
      }
      if (ss + 1 < nss) {
#pragma unroll
        for (int mb = 0; mb < kM; ++mb) {
          load_mb(mb, mb, ss + 1);
        }
      }
    }

    // Remainder: chunk i of the quad's last 128-code chunks is split into four 32-code units, unit q to lane q & 3.
    for (int i = 0; i < rem; ++i) {
      const int q = 4 * i + t;
      uint32_t rwe[kM][3];
      uint32_t rwo[kM][3];
      uint32_t rse[kM];
      uint32_t rso[kM];
#pragma unroll
      for (int mb = 0; mb < kM; ++mb) {
        const uint8_t* pe = wptr - 48 * t + static_cast<int64_t>(mb) * 16 * row_bytes + nss * 192 + 12 * q;
        const uint8_t* po = pe + row_bytes;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          rwe[mb][c] = LoadWord(pe + 4 * c);
          rwo[mb][c] = LoadWord(po + 4 * c);
        }
        const __nv_bfloat16* sb = sptr - 2 * t + static_cast<int64_t>(mb) * 16 * nb + 8 * nss + (q >> 1);
        const uint32_t s_e = LoadHalfWord(sb);
        const uint32_t s_o = LoadHalfWord(sb + nb);
        rse[mb] = s_e | (s_e << 16);
        rso[mb] = s_o | (s_o << 16);
      }
      const uint8_t* b_rem = smem + 8192 * nss + 2048 * i + 512 * t + ((g + 2 * t) & 7) * 16;
      uint4 bp[4];
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        bp[a] = *reinterpret_cast<const uint4*>(b_rem + a * 128);
      }
#pragma unroll
      for (int mb = 0; mb < kM; ++mb) {
        uint32_t se_mul, se_add, so_mul, so_add;
        DequantOperands(rse[mb], se_mul, se_add);
        DequantOperands(rso[mb], so_mul, so_add);
        GroupMma<0>(acc[mb], rwe[mb], rwo[mb], se_mul, se_add, so_mul, so_add, bp[0]);
        GroupMma<1>(acc[mb], rwe[mb], rwo[mb], se_mul, se_add, so_mul, so_add, bp[1]);
        GroupMma<2>(acc[mb], rwe[mb], rwo[mb], se_mul, se_add, so_mul, so_add, bp[2]);
        GroupMma<3>(acc[mb], rwe[mb], rwo[mb], se_mul, se_add, so_mul, so_add, bp[3]);
      }
    }

#pragma unroll
    for (int mb = 0; mb < kM; ++mb) {
      finalize(make_float4(acc[mb][0], acc[mb][1], acc[mb][2], acc[mb][3]), warp * kM + mb, lane);
    }
  }
#endif
}

inline bool DeviceSupportsBf16Mma() {
  int device = 0;
  int major = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
    return false;
  }
  return major >= 8;
}

// FC1: every warp owns kFc1M m-blocks (16 real rows each, gate/up interleaved) over the whole K, a CTA covers
// kFc1Warps * kFc1M * 16 rows. Every CTA stages the full activation rows, so tall tiles amortize that work.
constexpr int kFc1Warps = 4;
constexpr int kFc1M = 2;
constexpr int kFc1MinCtas = 3;
constexpr int kFc1TileN = kFc1Warps * kFc1M * 16;

// FC2: 4 warps each own kFc2M * 16 output rows over the whole (short) K.
constexpr int kFc2Warps = 4;
constexpr int kFc2M = 1;
constexpr int kFc2MinCtas = 5;
constexpr int kFc2TileN = kFc2Warps * kFc2M * 16;

// The dispatcher guarantees intermediate % 128 == 0 (FC1 n = 2 * intermediate) and hidden % kStepK == 0 (FC2 n).
static_assert((2 * 128) % kFc1TileN == 0 && kStepK % kFc2TileN == 0, "the tiles must divide the supported sizes");

inline void EnsureInt3TcCarveout() {
  static const bool carveout_set = []() {
    (void)cudaFuncSetAttribute(Int3TcKernel<true, kFc1Warps, kFc1M, kFc1MinCtas>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               cudaSharedmemCarveoutMaxShared);
    (void)cudaFuncSetAttribute(Int3TcKernel<false, kFc2Warps, kFc2M, kFc2MinCtas>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               cudaSharedmemCarveoutMaxShared);
    (void)cudaGetLastError();
    return true;
  }();
  (void)carveout_set;
}

inline cudaError_t LaunchFc1(const Params& fc1, int hidden, int grid_x, cudaStream_t stream) {
  EnsureInt3TcCarveout();
  const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(fc1.n / kFc1TileN));
  Int3TcKernel<true, kFc1Warps, kFc1M, kFc1MinCtas>
      <<<grid, kFc1Warps * 32, static_cast<size_t>(hidden) * 16, stream>>>(fc1);
  return cudaGetLastError();
}

inline cudaError_t LaunchFc2(const Params& fc2, int intermediate, int grid_x, cudaStream_t stream) {
  EnsureInt3TcCarveout();
  const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(fc2.n / kFc2TileN));
  Int3TcKernel<false, kFc2Warps, kFc2M, kFc2MinCtas>
      <<<grid, kFc2Warps * 32, static_cast<size_t>(intermediate) * 16, stream>>>(fc2);
  return cudaGetLastError();
}

// Returns true when the tensor-core kernels (FC1 and FC2) were launched, leaving the routes' finalize to the
// caller; false leaves the work to the vectorized GEMV kernels. `status` carries the launch result when handled.
inline bool TryLaunchBf16(const Int3MoeParams& params, cudaStream_t stream, cudaError_t& status) {
  const int64_t total_routes = (params.total_routes_override > 0)
                                   ? static_cast<int64_t>(params.total_routes_override)
                                   : static_cast<int64_t>(params.rows) * params.top_k;
  if (!params.is_bfloat16 || params.fc1.bits != 3 || params.fc2.bits != 3 || params.block_size != kBlockSize ||
      params.router_weights != nullptr || total_routes < params.tensor_core_min_routes ||
      total_routes > kMaxTotalRoutes ||
      params.hidden % kStepK != 0 || params.intermediate % 128 != 0 ||
      static_cast<int64_t>(params.hidden) * 16 > kMaxSharedBytes ||
      static_cast<int64_t>(params.intermediate) * 16 > kMaxSharedBytes ||
      (reinterpret_cast<uintptr_t>(params.fc1.weights) & 15u) != 0 ||
      (reinterpret_cast<uintptr_t>(params.fc2.weights) & 15u) != 0 ||
      (reinterpret_cast<uintptr_t>(params.fc1.scales) & 3u) != 0 ||
      (reinterpret_cast<uintptr_t>(params.fc2.scales) & 3u) != 0 ||
      params.fc1.zeros != nullptr || params.fc2.zeros != nullptr || params.experts == nullptr ||
      params.tensor_core_min_routes == 0 || !DeviceSupportsBf16Mma()) {
    return false;
  }

  // A build whose PTX is older than compute_80 holds empty stubs for the kernels.
  cudaFuncAttributes fc1_attributes{}, fc2_attributes{};
  if (cudaFuncGetAttributes(&fc1_attributes, Int3TcKernel<true, kFc1Warps, kFc1M, kFc1MinCtas>) != cudaSuccess ||
      cudaFuncGetAttributes(&fc2_attributes, Int3TcKernel<false, kFc2Warps, kFc2M, kFc2MinCtas>) != cudaSuccess ||
      fc1_attributes.ptxVersion < 80 || fc2_attributes.ptxVersion < 80) {
    return false;
  }

  Params fc1{};
  fc1.input = params.input;
  fc1.out = params.activation;
  fc1.weights = params.fc1.weights;
  fc1.scales = static_cast<const __nv_bfloat16*>(params.fc1.scales);
  fc1.bias = static_cast<const __nv_bfloat16*>(params.fc1.bias);
  fc1.experts = params.experts;
  fc1.source_route_ids = params.source_route_ids;
  fc1.input_is_gathered = params.input_is_gathered;
  fc1.total_routes = static_cast<int>(total_routes);
  fc1.num_experts = params.num_experts;
  fc1.top_k = params.top_k;
  fc1.n = 2 * params.intermediate;
  fc1.k = params.hidden;
  fc1.alpha = params.alpha;
  fc1.beta = params.beta;
  fc1.limit = params.limit;

  status = LaunchFc1(fc1, params.hidden, static_cast<int>(total_routes), stream);
  if (status != cudaSuccess) {
    return true;
  }

  Params fc2{};
  fc2.input = params.activation;
  fc2.out = params.fc2_partial;
  fc2.weights = params.fc2.weights;
  fc2.scales = static_cast<const __nv_bfloat16*>(params.fc2.scales);
  fc2.bias = static_cast<const __nv_bfloat16*>(params.fc2.bias);
  fc2.experts = params.experts;
  fc2.total_routes = static_cast<int>(total_routes);
  fc2.num_experts = params.num_experts;
  fc2.top_k = params.top_k;
  fc2.n = params.hidden;
  fc2.k = params.intermediate;

  status = LaunchFc2(fc2, params.intermediate, static_cast<int>(total_routes), stream);
  return true;
}

inline bool TryLaunchBf16Grouped(const Int3MoeParams& params, const int* sorted_routes,
                                 const Int3ExpertTileDesc* tile_descs, const int* num_tiles_ptr,
                                 int max_tiles, cudaStream_t stream, cudaError_t& status) {
  const int64_t total_routes = (params.total_routes_override > 0)
                                   ? static_cast<int64_t>(params.total_routes_override)
                                   : static_cast<int64_t>(params.rows) * params.top_k;
  if (!params.is_bfloat16 || params.fc1.bits != 3 || params.fc2.bits != 3 || params.block_size != kBlockSize ||
      params.router_weights != nullptr || max_tiles <= 0 || max_tiles > 65535 ||
      params.hidden % kStepK != 0 || params.intermediate % 128 != 0 ||
      static_cast<int64_t>(params.hidden) * 16 > kMaxSharedBytes ||
      static_cast<int64_t>(params.intermediate) * 16 > kMaxSharedBytes ||
      (reinterpret_cast<uintptr_t>(params.fc1.weights) & 15u) != 0 ||
      (reinterpret_cast<uintptr_t>(params.fc2.weights) & 15u) != 0 ||
      (reinterpret_cast<uintptr_t>(params.fc1.scales) & 3u) != 0 ||
      (reinterpret_cast<uintptr_t>(params.fc2.scales) & 3u) != 0 ||
      params.fc1.zeros != nullptr || params.fc2.zeros != nullptr ||
      sorted_routes == nullptr || tile_descs == nullptr || num_tiles_ptr == nullptr ||
      params.tensor_core_min_routes == 0 || !DeviceSupportsBf16Mma()) {
    return false;
  }

  cudaFuncAttributes fc1_attributes{}, fc2_attributes{};
  if (cudaFuncGetAttributes(&fc1_attributes, Int3TcKernel<true, kFc1Warps, kFc1M, kFc1MinCtas>) != cudaSuccess ||
      cudaFuncGetAttributes(&fc2_attributes, Int3TcKernel<false, kFc2Warps, kFc2M, kFc2MinCtas>) != cudaSuccess ||
      fc1_attributes.ptxVersion < 80 || fc2_attributes.ptxVersion < 80) {
    return false;
  }

  Params fc1{};
  fc1.input = params.input;
  fc1.out = params.activation;
  fc1.weights = params.fc1.weights;
  fc1.scales = static_cast<const __nv_bfloat16*>(params.fc1.scales);
  fc1.bias = static_cast<const __nv_bfloat16*>(params.fc1.bias);
  fc1.experts = params.experts;
  fc1.source_route_ids = params.source_route_ids;
  fc1.sorted_routes = sorted_routes;
  fc1.tile_descs = tile_descs;
  fc1.num_tiles_ptr = num_tiles_ptr;
  fc1.input_is_gathered = params.input_is_gathered;
  fc1.total_routes = static_cast<int>(total_routes);
  fc1.num_experts = params.num_experts;
  fc1.top_k = params.top_k;
  fc1.n = 2 * params.intermediate;
  fc1.k = params.hidden;
  fc1.alpha = params.alpha;
  fc1.beta = params.beta;
  fc1.limit = params.limit;

  status = LaunchFc1(fc1, params.hidden, max_tiles, stream);
  if (status != cudaSuccess) {
    return true;
  }

  Params fc2{};
  fc2.input = params.activation;
  fc2.out = params.fc2_partial;
  fc2.weights = params.fc2.weights;
  fc2.scales = static_cast<const __nv_bfloat16*>(params.fc2.scales);
  fc2.bias = static_cast<const __nv_bfloat16*>(params.fc2.bias);
  fc2.experts = params.experts;
  fc2.sorted_routes = sorted_routes;
  fc2.tile_descs = tile_descs;
  fc2.num_tiles_ptr = num_tiles_ptr;
  fc2.total_routes = static_cast<int>(total_routes);
  fc2.num_experts = params.num_experts;
  fc2.top_k = params.top_k;
  fc2.n = params.hidden;
  fc2.k = params.intermediate;

  status = LaunchFc2(fc2, params.intermediate, max_tiles, stream);
  return true;
}

#endif  // __CUDACC__

}  // namespace int3_tc
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
