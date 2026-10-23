// Licensed under the MIT License.
// Arithmetic-preserving port of tianleiwu/qwen38-flash-ops src/mtp_head/qwen_mtp_ops.cu.
// Only routing and quantized expert kernels are retained; ORT validation/registration is native.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace onnxruntime::contrib::cuda::mtp_moe {
using bf16 = __nv_bfloat16;
constexpr int kHidden = 2560, kExperts = 512, kTopK = 10, kFfn = 640;
constexpr int kGateUp = 2 * kFfn, kChunk = 8, kScratchStride = 32 + kTopK * kFfn;
constexpr int kWarpsPerBlock = 8, kKChunks = kHidden / kChunk, kPerLane = kKChunks / 32;
constexpr int kDownChunks = kFfn / kChunk, kDownPerLane = (kDownChunks + 31) / 32;
__device__ __forceinline__ float R(float v) { return __bfloat162float(__float2bfloat16(v)); }

__device__ __forceinline__ float Sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }

__device__ __forceinline__ void Unpack(const uint4& v, float* f) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ void UnpackFp8(const uint2& v, float* f) {
  const uint32_t w[2] = {v.x, v.y};
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const __half2_raw hr01 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w[i] & 0xffffu), __NV_E4M3);
    const __half2_raw hr23 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w[i] >> 16), __NV_E4M3);
    const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&hr01));
    const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&hr23));
    f[4 * i + 0] = f01.x;
    f[4 * i + 1] = f01.y;
    f[4 * i + 2] = f23.x;
    f[4 * i + 3] = f23.y;
  }
}

__device__ __forceinline__ float Fp8ByteToFloat(uint8_t b) {
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
  return __half2float(*reinterpret_cast<const __half*>(&hr));
}

__device__ __forceinline__ float E2M1ToFloat(uint32_t nib) {
  const uint32_t m = nib & 7u;
  const uint32_t exp_mant = (m == 0u) ? 0u : ((m == 1u ? 252u : (m + 252u)) << 22);
  return __uint_as_float(((nib & 8u) << 28) | exp_mant);
}

__device__ __forceinline__ void UnpackFp4(uint32_t pack8, uint8_t scale_fp8, float* f) {
  const float s_half = Fp8ByteToFloat(scale_fp8) * 0.5f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t nib = (pack8 >> (4 * i)) & 0xfu;
    const uint32_t mag2 = (0xC8643210u >> ((nib & 7u) * 4u)) & 0xfu;
    const float v = static_cast<float>(mag2) * s_half;
    f[i] = __uint_as_float(__float_as_uint(v) ^ ((nib & 8u) << 28));
  }
}

__global__ void RouterKernel(const bf16* __restrict__ logits, float* __restrict__ scratch) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  constexpr int kIter = kExperts / 32;
  float v[kIter];
  float mx = -INFINITY;
#pragma unroll
  for (int i = 0; i < kIter; ++i) {
    v[i] = __bfloat162float(logits[static_cast<int64_t>(row) * kExperts + i * 32 + lane]);
    mx = fmaxf(mx, v[i]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, offset));
  float e[kIter];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kIter; ++i) {
    e[i] = expf(v[i] - mx);
    sum += e[i];
  }
  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  float p[kIter];
#pragma unroll
  for (int i = 0; i < kIter; ++i) p[i] = e[i] / sum;

  float top_v[kTopK];
  int top_i[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float best = -1.0f;
    int best_i = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < kIter; ++i) {
      if (p[i] > best) {
        best = p[i];
        best_i = i * 32 + lane;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, offset);
      const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
      if (ov > best || (ov == best && oi < best_i)) {
        best = ov;
        best_i = oi;
      }
    }
    if (best_i < 0 || best_i >= kExperts) {
      best = 0.0f;
      best_i = k;
    }
    top_v[k] = best;
    top_i[k] = best_i;
    // remove the winner from its owner lane
#pragma unroll
    for (int i = 0; i < kIter; ++i) {
      if (best_i == i * 32 + lane) p[i] = -2.0f;
    }
  }
  if (lane == 0) {
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) total += top_v[k];
    float* out = scratch + static_cast<int64_t>(row) * kScratchStride;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      out[k] = __int_as_float(top_i[k]);
      out[kTopK + k] = (total > 0.0f) ? R(top_v[k] / total) : 0.0f;
    }
  }
}

template <int MODE>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    GateUpQuantKernel(const bf16* __restrict__ x, const uint8_t* __restrict__ gate_up, float* __restrict__ scratch,
                      float global_scale) {
  constexpr int kRowBytes = MODE == 1 ? (kHidden + 16) : (kHidden * 9 / 16);
  const int row = blockIdx.z;
  const int slot = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float* base = scratch + static_cast<int64_t>(row) * kScratchStride;
  const int expert = __float_as_int(base[slot]);
  const int j0 = (blockIdx.x * kWarpsPerBlock + warp) * 2;
  const uint4* xv = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(row) * kHidden);
  const uint8_t* rg0 = gate_up + (static_cast<int64_t>(expert) * kGateUp + j0) * kRowBytes;
  const uint8_t* rg1 = rg0 + kRowBytes;
  const uint8_t* ru0 = gate_up + (static_cast<int64_t>(expert) * kGateUp + kFfn + j0) * kRowBytes;
  const uint8_t* ru1 = ru0 + kRowBytes;
  uint4 xr[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    xr[i] = __ldg(xv + i * 32 + lane);
  }
  float acc_g0 = 0.0f, acc_u0 = 0.0f, acc_g1 = 0.0f, acc_u1 = 0.0f;
  if constexpr (MODE == 1) {
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int c = i * 32 + lane;
      const uint2 g20 = __ldg(reinterpret_cast<const uint2*>(rg0) + c);
      const uint2 u20 = __ldg(reinterpret_cast<const uint2*>(ru0) + c);
      const uint2 g21 = __ldg(reinterpret_cast<const uint2*>(rg1) + c);
      const uint2 u21 = __ldg(reinterpret_cast<const uint2*>(ru1) + c);
      float fg0[8], fu0[8], fg1[8], fu1[8], fx[8];
      UnpackFp8(g20, fg0);
      UnpackFp8(u20, fu0);
      UnpackFp8(g21, fg1);
      UnpackFp8(u21, fu1);
      Unpack(xr[i], fx);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        acc_g0 = fmaf(fg0[k], fx[k], acc_g0);
        acc_u0 = fmaf(fu0[k], fx[k], acc_u0);
        acc_g1 = fmaf(fg1[k], fx[k], acc_g1);
        acc_u1 = fmaf(fu1[k], fx[k], acc_u1);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int c = i * 32 + lane;
      const uint32_t g40 = __ldg(reinterpret_cast<const uint32_t*>(rg0) + c);
      const uint32_t u40 = __ldg(reinterpret_cast<const uint32_t*>(ru0) + c);
      const uint32_t g41 = __ldg(reinterpret_cast<const uint32_t*>(rg1) + c);
      const uint32_t u41 = __ldg(reinterpret_cast<const uint32_t*>(ru1) + c);
      const uint8_t gs0 = __ldg(rg0 + (kHidden / 2) + (c >> 1));
      const uint8_t us0 = __ldg(ru0 + (kHidden / 2) + (c >> 1));
      const uint8_t gs1 = __ldg(rg1 + (kHidden / 2) + (c >> 1));
      const uint8_t us1 = __ldg(ru1 + (kHidden / 2) + (c >> 1));
      float fg0[8], fu0[8], fg1[8], fu1[8], fx[8];
      UnpackFp4(g40, gs0, fg0);
      UnpackFp4(u40, us0, fu0);
      UnpackFp4(g41, gs1, fg1);
      UnpackFp4(u41, us1, fu1);
      Unpack(xr[i], fx);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        acc_g0 = fmaf(fg0[k], fx[k], acc_g0);
        acc_u0 = fmaf(fu0[k], fx[k], acc_u0);
        acc_g1 = fmaf(fg1[k], fx[k], acc_g1);
        acc_u1 = fmaf(fu1[k], fx[k], acc_u1);
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc_g0 += __shfl_xor_sync(0xffffffffu, acc_g0, offset);
    acc_u0 += __shfl_xor_sync(0xffffffffu, acc_u0, offset);
    acc_g1 += __shfl_xor_sync(0xffffffffu, acc_g1, offset);
    acc_u1 += __shfl_xor_sync(0xffffffffu, acc_u1, offset);
  }
  if (lane == 0) {
    if constexpr (MODE == 1) {
      acc_g0 *= *reinterpret_cast<const float*>(rg0 + kHidden);
      acc_u0 *= *reinterpret_cast<const float*>(ru0 + kHidden);
      acc_g1 *= *reinterpret_cast<const float*>(rg1 + kHidden);
      acc_u1 *= *reinterpret_cast<const float*>(ru1 + kHidden);
    } else {
      acc_g0 *= global_scale;
      acc_u0 *= global_scale;
      acc_g1 *= global_scale;
      acc_u1 *= global_scale;
    }
    const float gv0 = R(acc_g0), uv0 = R(acc_u0);
    const float gv1 = R(acc_g1), uv1 = R(acc_u1);
    const float s0 = R(gv0 * Sigmoid(gv0));
    const float s1 = R(gv1 * Sigmoid(gv1));
    bf16* dst = reinterpret_cast<bf16*>(base + 32) + slot * kFfn + j0;
    dst[0] = __float2bfloat16(s0 * uv0);
    dst[1] = __float2bfloat16(s1 * uv1);
  }
}

template <int MODE>
__global__ void __launch_bounds__(kTopK * 32)
    DownQuantKernel(const uint8_t* __restrict__ down, const float* __restrict__ scratch, bf16* __restrict__ out,
                    float global_scale) {
  constexpr int kRowBytes = MODE == 1 ? (kFfn + 16) : (kFfn * 9 / 16);
  const int row = blockIdx.y;
  const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int i0 = blockIdx.x * 4;
  const float* base = scratch + static_cast<int64_t>(row) * kScratchStride;
  const int expert = __float_as_int(base[k]);
  const uint8_t* rw0 = down + (static_cast<int64_t>(expert) * kHidden + i0) * kRowBytes;
  const uint4* h = reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(base + 32) + k * kFfn);
  float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  uint4 hv[kDownPerLane];
#pragma unroll
  for (int t = 0; t < kDownPerLane; ++t) {
    const int c = t * 32 + lane;
    hv[t] = c < kDownChunks ? __ldg(h + c) : make_uint4(0, 0, 0, 0);
  }
  if constexpr (MODE == 1) {
#pragma unroll
    for (int t = 0; t < kDownPerLane; ++t) {
      const int c = t * 32 + lane;
      if (c < kDownChunks) {
        float fh[8];
        Unpack(hv[t], fh);
#pragma unroll
        for (int col = 0; col < 4; ++col) {
          const uint2 w2 = __ldg(reinterpret_cast<const uint2*>(rw0 + col * kRowBytes) + c);
          float fw[8];
          UnpackFp8(w2, fw);
#pragma unroll
          for (int e = 0; e < 8; ++e) sum[col] = fmaf(fw[e], fh[e], sum[col]);
        }
      }
    }
  } else {
#pragma unroll
    for (int t = 0; t < kDownPerLane; ++t) {
      const int c = t * 32 + lane;
      if (c < kDownChunks) {
        float fh[8];
        Unpack(hv[t], fh);
#pragma unroll
        for (int col = 0; col < 4; ++col) {
          const uint8_t* rw = rw0 + col * kRowBytes;
          const uint32_t w4 = __ldg(reinterpret_cast<const uint32_t*>(rw) + c);
          const uint8_t ws = __ldg(rw + (kFfn / 2) + (c >> 1));
          float fw[8];
          UnpackFp4(w4, ws, fw);
#pragma unroll
          for (int e = 0; e < 8; ++e) sum[col] = fmaf(fw[e], fh[e], sum[col]);
        }
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int col = 0; col < 4; ++col) {
      sum[col] += __shfl_xor_sync(0xffffffffu, sum[col], offset);
    }
  }
  __shared__ float weighted[kTopK][4];
  if (lane == 0) {
    const float w_k = base[kTopK + k];
#pragma unroll
    for (int col = 0; col < 4; ++col) {
      float s = sum[col];
      if constexpr (MODE == 1) {
        s *= *reinterpret_cast<const float*>(rw0 + col * kRowBytes + kFfn);
      } else {
        s *= global_scale;
      }
      weighted[k][col] = R(R(s) * w_k);
    }
  }
  __syncthreads();
  if (threadIdx.x < 4) {
    const int col = threadIdx.x;
    float total = 0.0f;
#pragma unroll
    for (int e = 0; e < kTopK; ++e) total += weighted[e][col];
    out[static_cast<int64_t>(row) * kHidden + i0 + col] = __float2bfloat16(total);
  }
}
}  // namespace onnxruntime::contrib::cuda::mtp_moe
