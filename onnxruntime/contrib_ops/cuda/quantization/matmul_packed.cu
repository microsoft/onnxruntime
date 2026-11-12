// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_packed.h"

#include <algorithm>
#include <gsl/util>
#include <type_traits>

#include "contrib_ops/cuda/quantization/gguf_quant_cuda.cuh"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Warp-K-Split Tensor-Core BF16 GEMV/GEMM kernel for M in 1..8 on large-N projections (e.g. lm_head).
// KW warps (KW*32 threads) split K across the KW warps with zero __syncthreads() in the K-loop,
// while computing MT*16 output columns per CTA.
template <int KW, int MT, int kCompileType, int kChunk = 32>
__global__ __launch_bounds__(KW * 32) void MatMulPackedBf16TcWarpKSplitKernel(
    const __nv_bfloat16* __restrict__ a,
    MatMulPackedParams p,
    __nv_bfloat16* __restrict__ y,
    int m_rt) {
  constexpr int kThreads = KW * 32;
  __shared__ float s_red[KW][MT * 16][8];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;  // 0..7
  const int t = lane & 3;   // 0..3

  const int n_out_base = static_cast<int>(blockIdx.x) * (MT * 16);
  const uint8_t* row_ptrs_lo[MT];
  const uint8_t* row_ptrs_hi[MT];
  bool valid_lo[MT];
  bool valid_hi[MT];
  float acc[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    const int r_lo = n_out_base + m * 16 + g;
    const int r_hi = r_lo + 8;
    valid_lo[m] = (r_lo < p.n);
    valid_hi[m] = (r_hi < p.n);
    row_ptrs_lo[m] = p.b_data + static_cast<int64_t>(valid_lo[m] ? r_lo : 0) * p.row_bytes;
    row_ptrs_hi[m] = p.b_data + static_cast<int64_t>(valid_hi[m] ? r_hi : 0) * p.row_bytes;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc[m][i] = 0.0f;
    }
  }

  const bool a_valid = (g < m_rt);
  const bool a_aligned = ((reinterpret_cast<uintptr_t>(a) & 15u) == 0);
  const __nv_bfloat16* a_row = a + static_cast<int64_t>(a_valid ? g : 0) * p.k;

  if constexpr (kChunk != 32) {
    // Keep contiguous 64/128-K partitions and their MMA operand order together;
    // interleaving 32-K partitions changes cancellation before the warp reduction.
    constexpr int kHalves = kChunk / 64;
    for (int chunk = warp; chunk < p.k / kChunk; chunk += KW) {
#pragma unroll
      for (int half = 0; half < kHalves; ++half) {
        const int offset = chunk * kChunk + t * (kChunk / 4) + half * 16;
        uint4 x0 = make_uint4(0u, 0u, 0u, 0u);
        uint4 x1 = x0;
        if (a_valid) {
          const auto* input = a_row + offset;
          if ((reinterpret_cast<uintptr_t>(input) & 15u) == 0) {
            x0 = __ldg(reinterpret_cast<const uint4*>(input));
            x1 = __ldg(reinterpret_cast<const uint4*>(input + 8));
          } else {
            const auto* bits = reinterpret_cast<const uint16_t*>(input);
            auto* low = reinterpret_cast<uint32_t*>(&x0);
            auto* high = reinterpret_cast<uint32_t*>(&x1);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              low[i] = __ldg(bits + 2 * i) | (static_cast<uint32_t>(__ldg(bits + 2 * i + 1)) << 16);
              high[i] = __ldg(bits + 8 + 2 * i) | (static_cast<uint32_t>(__ldg(bits + 9 + 2 * i)) << 16);
            }
          }
        }
        const uint32_t x_words[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          float w_lo[16] = {};
          float w_hi[16] = {};
          if (valid_lo[m]) {
            gguf_cuda::DecodeGgufSubBlock16Impl<kCompileType>(
                row_ptrs_lo[m], offset / 32, (offset / 16) & 1, w_lo);
          }
          if (valid_hi[m]) {
            gguf_cuda::DecodeGgufSubBlock16Impl<kCompileType>(
                row_ptrs_hi[m], offset / 32, (offset / 16) & 1, w_hi);
          }
          float e0[8], e1[8], o0[8], o1[8];
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            constexpr bool adjacent = kCompileType == gguf_quant::kGgmlTypeIQ2_XXS;
            const int i0 = adjacent ? 4 * s : s;
            const int i1 = adjacent ? i0 + 1 : s + 4;
            const int i2 = adjacent ? i0 + 2 : s + 8;
            const int i3 = adjacent ? i0 + 3 : s + 12;
            e0[2 * s] = w_lo[i0];
            e0[2 * s + 1] = w_lo[i1];
            e1[2 * s] = w_lo[i2];
            e1[2 * s + 1] = w_lo[i3];
            o0[2 * s] = w_hi[i0];
            o0[2 * s + 1] = w_hi[i1];
            o1[2 * s] = w_hi[i2];
            o1[2 * s + 1] = w_hi[i3];
          }
          const uint4 e_lo = gguf_cuda::PackFloat8ToBf16x2(e0);
          const uint4 e_hi = gguf_cuda::PackFloat8ToBf16x2(e1);
          const uint4 o_lo = gguf_cuda::PackFloat8ToBf16x2(o0);
          const uint4 o_hi = gguf_cuda::PackFloat8ToBf16x2(o1);
          const uint32_t e_low[4] = {e_lo.x, e_lo.y, e_lo.z, e_lo.w};
          const uint32_t e_high[4] = {e_hi.x, e_hi.y, e_hi.z, e_hi.w};
          const uint32_t o_low[4] = {o_lo.x, o_lo.y, o_lo.z, o_lo.w};
          const uint32_t o_high[4] = {o_hi.x, o_hi.y, o_hi.z, o_hi.w};
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            uint32_t b0, b1;
            if constexpr (kCompileType == gguf_quant::kGgmlTypeIQ2_XXS) {
              b0 = x_words[2 * s];
              b1 = x_words[2 * s + 1];
            } else {
              const int word = s / 2;
              const int selector = (s & 1) ? 0x7632 : 0x5410;
              b0 = __byte_perm(x_words[word], x_words[word + 2], selector);
              b1 = __byte_perm(x_words[word + 4], x_words[word + 6], selector);
            }
            gguf_cuda::MmaBf16(acc[m], e_low[s], o_low[s], e_high[s], o_high[s], b0, b1);
          }
        }
      }
    }
  } else {
    const int num_sb32 = p.k >> 5;
    for (int sb32_idx = warp; sb32_idx < num_sb32; sb32_idx += KW) {
      const __nv_bfloat16* a_ptr = a_row + sb32_idx * 32 + t * 8;
      const uint4 b_vec = a_valid
                              ? (a_aligned ? __ldg(reinterpret_cast<const uint4*>(a_ptr))
                                           : gguf_cuda::LoadU4Dev(a_ptr))
                              : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        float w_lo[8] = {};
        float w_hi[8] = {};
        if (valid_lo[m]) {
          gguf_cuda::DecodeGgufSubBlock8Tpl<kCompileType>(row_ptrs_lo[m], p.ggml_type, sb32_idx, t, w_lo);
        }
        if (valid_hi[m]) {
          gguf_cuda::DecodeGgufSubBlock8Tpl<kCompileType>(row_ptrs_hi[m], p.ggml_type, sb32_idx, t, w_hi);
        }
        // Warp-synchronous MMA requires every lane, including padded output rows.
        const uint4 e_vec = gguf_cuda::PackFloat8ToBf16x2(w_lo);
        const uint4 o_vec = gguf_cuda::PackFloat8ToBf16x2(w_hi);
        gguf_cuda::MmaBf16(acc[m], e_vec.x, o_vec.x, e_vec.y, o_vec.y, b_vec.x, b_vec.y);
        gguf_cuda::MmaBf16(acc[m], e_vec.z, o_vec.z, e_vec.w, o_vec.w, b_vec.z, b_vec.w);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < MT; ++m) {
    s_red[warp][m * 16 + g][2 * t + 0] = acc[m][0];
    s_red[warp][m * 16 + g][2 * t + 1] = acc[m][1];
    s_red[warp][m * 16 + g + 8][2 * t + 0] = acc[m][2];
    s_red[warp][m * 16 + g + 8][2 * t + 1] = acc[m][3];
  }
  __syncthreads();

  for (int idx = tid; idx < MT * 16 * 8; idx += kThreads) {
    const int row_m = idx / (MT * 16);
    const int rl = idx - row_m * (MT * 16);
    const int col = n_out_base + rl;
    if (row_m < m_rt && col < p.n) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < KW; ++w) {
        sum += s_red[w][rl][row_m];
      }
      y[static_cast<int64_t>(row_m) * p.n + col] =
          __float2bfloat16_rn(p.global_scale ? sum * *p.global_scale : sum);
    }
  }
}

template <typename CudaT, int M_FIXED, int NR, int KW, int kCompileType>
__global__ __launch_bounds__(256) void MatMulPackedGemvKernel(
    const CudaT* __restrict__ a,
    MatMulPackedParams p,
    CudaT* __restrict__ y,
    int m_rt) {
  constexpr int kWarpsPerBlock = 8;
  constexpr int kGroupsPerBlock = kWarpsPerBlock / KW;
  constexpr int kMaxM = (M_FIXED > 0) ? M_FIXED : 8;
  __shared__ float s_red[kGroupsPerBlock][(KW > 1) ? KW : 1][NR][kMaxM];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int grp = warp / KW;
  const int warp_k = warp % KW;
  const int n_base = (static_cast<int>(blockIdx.x) * kGroupsPerBlock + grp) * NR;
  const bool valid_group = (n_base < p.n);
  if constexpr (KW == 1) {
    if (!valid_group) return;
  }

  const int M = M_FIXED > 0 ? M_FIXED : m_rt;
  const bool a_aligned = ((reinterpret_cast<uintptr_t>(a) & 15u) == 0);

  const uint8_t* row_ptrs[NR];
  if (valid_group) {
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      const int n_out = (n_base + r < p.n) ? (n_base + r) : (p.n - 1);
      row_ptrs[r] = p.b_data + static_cast<int64_t>(n_out) * p.row_bytes;
    }
  }

  float acc[NR][kMaxM];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
#pragma unroll
    for (int m = 0; m < kMaxM; ++m) {
      acc[r][m] = 0.0f;
    }
  }

  const int total_quarters = p.k >> 3;
  if (valid_group) {
#pragma unroll 1
    for (int q_idx = warp_k * 32 + lane; q_idx < total_quarters; q_idx += KW * 32) {
      const int sb32_idx = q_idx >> 2;
      const int l = q_idx & 3;
      float w_vals[NR][8];
#pragma unroll
      for (int r = 0; r < NR; ++r) {
        gguf_cuda::DecodeGgufSubBlock8Tpl<kCompileType>(row_ptrs[r], p.ggml_type, sb32_idx, l, w_vals[r]);
        if (!p.allow_unrounded_gemv ||
            (std::is_same_v<CudaT, __nv_bfloat16> && p.n >= kMatMulPackedBf16TcMinN)) {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            w_vals[r][j] = gguf_cuda::CudaTypeToFloat(
                gguf_cuda::FloatToCudaType<CudaT>(w_vals[r][j]));
          }
        }
      }
      const int k_offset = sb32_idx * 32 + (l << 3);

      if constexpr (M_FIXED > 0) {
        if (a_aligned) {
#pragma unroll
          for (int m = 0; m < M_FIXED; ++m) {
            const CudaT* a_ptr = a + static_cast<int64_t>(m) * p.k + k_offset;
            const uint4 v0 = __ldg(reinterpret_cast<const uint4*>(a_ptr));
            const CudaT* a0 = reinterpret_cast<const CudaT*>(&v0);
            float af[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              af[j] = gguf_cuda::CudaTypeToFloat<CudaT>(a0[j]);
            }
#pragma unroll
            for (int r = 0; r < NR; ++r) {
#pragma unroll
              for (int j = 0; j < 8; ++j) {
                acc[r][m] += w_vals[r][j] * af[j];
              }
            }
          }
        } else {
#pragma unroll
          for (int m = 0; m < M_FIXED; ++m) {
            const CudaT* a_ptr = a + static_cast<int64_t>(m) * p.k + k_offset;
            const uint4 v0 = gguf_cuda::LoadU4Dev(a_ptr);
            const CudaT* a0 = reinterpret_cast<const CudaT*>(&v0);
            float af[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              af[j] = gguf_cuda::CudaTypeToFloat<CudaT>(a0[j]);
            }
#pragma unroll
            for (int r = 0; r < NR; ++r) {
#pragma unroll
              for (int j = 0; j < 8; ++j) {
                acc[r][m] += w_vals[r][j] * af[j];
              }
            }
          }
        }
      } else {
        for (int m = 0; m < M; ++m) {
          const CudaT* a_ptr = a + static_cast<int64_t>(m) * p.k + k_offset;
          const uint4 v0 = a_aligned ? __ldg(reinterpret_cast<const uint4*>(a_ptr)) : gguf_cuda::LoadU4Dev(a_ptr);
          const CudaT* a0 = reinterpret_cast<const CudaT*>(&v0);
          float af[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            af[j] = gguf_cuda::CudaTypeToFloat<CudaT>(a0[j]);
          }
#pragma unroll
          for (int r = 0; r < NR; ++r) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              acc[r][m] += w_vals[r][j] * af[j];
            }
          }
        }
      }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int r = 0; r < NR; ++r) {
#pragma unroll
        for (int m = 0; m < kMaxM; ++m) {
          acc[r][m] += __shfl_down_sync(0xFFFFFFFFu, acc[r][m], offset);
        }
      }
    }
  }

  if constexpr (KW == 1) {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < NR; ++r) {
        const int n_out = n_base + r;
        if (NR == 1 || n_out < p.n) {
          if constexpr (M_FIXED > 0) {
#pragma unroll
            for (int m = 0; m < M_FIXED; ++m) {
              y[static_cast<int64_t>(m) * p.n + n_out] =
                  gguf_cuda::FloatToCudaType<CudaT>(p.global_scale ? acc[r][m] * *p.global_scale : acc[r][m]);
            }
          } else {
            for (int m = 0; m < M; ++m) {
              y[static_cast<int64_t>(m) * p.n + n_out] =
                  gguf_cuda::FloatToCudaType<CudaT>(p.global_scale ? acc[r][m] * *p.global_scale : acc[r][m]);
            }
          }
        }
      }
    }
  } else {
    if (valid_group && lane == 0) {
#pragma unroll
      for (int r = 0; r < NR; ++r) {
#pragma unroll
        for (int m = 0; m < kMaxM; ++m) {
          s_red[grp][warp_k][r][m] = acc[r][m];
        }
      }
    }
    __syncthreads();
    if (valid_group && warp_k == 0 && lane == 0) {
#pragma unroll
      for (int r = 0; r < NR; ++r) {
        const int n_out = n_base + r;
        if (NR == 1 || n_out < p.n) {
          if constexpr (M_FIXED > 0) {
#pragma unroll
            for (int m = 0; m < M_FIXED; ++m) {
              float sum = 0.0f;
#pragma unroll
              for (int w = 0; w < KW; ++w) {
                sum += s_red[grp][w][r][m];
              }
              y[static_cast<int64_t>(m) * p.n + n_out] =
                  gguf_cuda::FloatToCudaType<CudaT>(p.global_scale ? sum * *p.global_scale : sum);
            }
          } else {
            for (int m = 0; m < M; ++m) {
              float sum = 0.0f;
#pragma unroll
              for (int w = 0; w < KW; ++w) {
                sum += s_red[grp][w][r][m];
              }
              y[static_cast<int64_t>(m) * p.n + n_out] =
                  gguf_cuda::FloatToCudaType<CudaT>(p.global_scale ? sum * *p.global_scale : sum);
            }
          }
        }
      }
    }
  }
}

template <typename CudaT, typename ActivationT = CudaT>
__global__ __launch_bounds__(256) void MatMulPackedDequantKernel(
    MatMulPackedParams p,
    CudaT* __restrict__ w_out) {
  constexpr int kWarpsPerBlock = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n_out = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
  if (n_out >= p.n) return;

  const uint8_t* row_ptr = p.b_data + static_cast<int64_t>(n_out) * p.row_bytes;
  CudaT* dst_row = w_out + static_cast<int64_t>(n_out) * p.k;

  const int total_quarters = p.k / 8;
  for (int q_idx = lane; q_idx < total_quarters; q_idx += 32) {
    const int sb32_idx = q_idx >> 2;
    const int l = q_idx & 3;
    float w_vals[8];
    gguf_cuda::DecodeGgufSubBlock8(row_ptr, p.ggml_type, sb32_idx, l, w_vals);
    CudaT* dst8 = dst_row + sb32_idx * 32 + l * 8;
    if constexpr (std::is_same_v<CudaT, float>) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        // FP32 scratch preserves the activation-dtype operand rounding used by GEMM.
        dst8[j] = gguf_cuda::CudaTypeToFloat(
            gguf_cuda::FloatToCudaType<ActivationT>(w_vals[j]));
      }
    } else if constexpr (std::is_same<CudaT, __nv_bfloat16>::value) {
      *reinterpret_cast<uint4*>(dst8) = gguf_cuda::PackFloat8ToBf16x2(w_vals);
    } else {
      uint4 out_u4;
      CudaT* out_elems = reinterpret_cast<CudaT*>(&out_u4);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        out_elems[j] = gguf_cuda::FloatToCudaType<CudaT>(w_vals[j]);
      }
      *reinterpret_cast<uint4*>(dst8) = out_u4;
    }
  }
}

template <typename SrcT, typename DstT>
__global__ void MatMulPackedCastTileKernel(
    const SrcT* src, DstT* dst, int columns, int src_stride, int dst_stride,
    const float* global_scale = nullptr) {
  const int row = static_cast<int>(blockIdx.y);
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < columns) {
    dst[static_cast<int64_t>(row) * dst_stride + col] =
        gguf_cuda::FloatToCudaType<DstT>(
            gguf_cuda::CudaTypeToFloat<SrcT>(src[static_cast<int64_t>(row) * src_stride + col]) *
            (global_scale ? *global_scale : 1.0f));
  }
}

Status LaunchMatMulPackedFp32Prefill(
    cublasHandle_t cublas, cudaStream_t stream, const half* a,
    const MatMulPackedParams& params, float* workspace, half* y, int m) {
  const int max_n = std::min(kMatMulPackedPrefillNTile, params.n);
  const int max_m = std::min(kMatMulPackedPrefillMTile, m);
  float* weights = workspace;
  float* activations = weights + static_cast<size_t>(SafeInt<size_t>(max_n) * params.k);
  float* output = activations + static_cast<size_t>(SafeInt<size_t>(max_m) * params.k);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  for (int n0 = 0; n0 < params.n;) {
    MatMulPackedParams tile_params = params;
    tile_params.n = std::min(max_n, params.n - n0);
    tile_params.b_data = params.b_data + static_cast<size_t>(SafeInt<size_t>(n0) * params.row_bytes);
    const dim3 weight_grid(static_cast<unsigned int>((tile_params.n + 7) / 8));
    MatMulPackedDequantKernel<float, half><<<weight_grid, 256, 0, stream>>>(tile_params, weights);
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
    for (int m0 = 0; m0 < m;) {
      const int tile_m = std::min(max_m, m - m0);
      const dim3 activation_grid((static_cast<unsigned int>(params.k) + 255) / 256,
                                 static_cast<unsigned int>(tile_m));
      MatMulPackedCastTileKernel<<<activation_grid, 256, 0, stream>>>(
          a + static_cast<int64_t>(m0) * params.k, activations, params.k, params.k, params.k);
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
      CUBLAS_RETURN_IF_ERROR(cublasGemmEx(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, tile_params.n, tile_m, params.k,
          &alpha, weights, CUDA_R_32F, params.k, activations, CUDA_R_32F, params.k,
          &beta, output, CUDA_R_32F, tile_params.n, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
      const dim3 output_grid((static_cast<unsigned int>(tile_params.n) + 255) / 256,
                             static_cast<unsigned int>(tile_m));
      MatMulPackedCastTileKernel<<<output_grid, 256, 0, stream>>>(
          output, y + static_cast<int64_t>(m0) * params.n + n0, tile_params.n, tile_params.n, params.n,
          params.global_scale);
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
      m0 += tile_m;
    }
    n0 += tile_params.n;
  }
  return Status::OK();
}

template <typename CudaT, int NR, int KW, int kCompileType>
void LaunchMatMulPackedGemvByMWithNR(
    cudaStream_t stream,
    const CudaT* a,
    const MatMulPackedParams& params,
    CudaT* y,
    int m) {
  constexpr int kWarpsPerBlock = 8;
  constexpr int kGroupsPerBlock = kWarpsPerBlock / KW;
  constexpr int kRowsPerBlock = kGroupsPerBlock * NR;
  const dim3 grid(static_cast<unsigned int>((params.n + kRowsPerBlock - 1) / kRowsPerBlock));
  const dim3 block(256);
  switch (m) {
    case 1:
      MatMulPackedGemvKernel<CudaT, 1, NR, KW, kCompileType><<<grid, block, 0, stream>>>(a, params, y, 1);
      return;
    case 2:
      MatMulPackedGemvKernel<CudaT, 2, NR, KW, kCompileType><<<grid, block, 0, stream>>>(a, params, y, 2);
      return;
    case 4:
      MatMulPackedGemvKernel<CudaT, 4, NR, KW, kCompileType><<<grid, block, 0, stream>>>(a, params, y, 4);
      return;
    case 7:
      MatMulPackedGemvKernel<CudaT, 7, NR, KW, kCompileType><<<grid, block, 0, stream>>>(a, params, y, 7);
      return;
    case 8:
      MatMulPackedGemvKernel<CudaT, 8, NR, KW, kCompileType><<<grid, block, 0, stream>>>(a, params, y, 8);
      return;
    default:
      MatMulPackedGemvKernel<CudaT, 0, NR, KW, kCompileType><<<grid, block, 0, stream>>>(a, params, y, m);
      return;
  }
}

template <typename CudaT, int kCompileType>
void LaunchMatMulPackedGemvByM(
    cudaStream_t stream,
    dim3 grid,
    const CudaT* a,
    const MatMulPackedParams& params,
    CudaT* y,
    int m) {
  (void)grid;
  if (m == 1) {
    LaunchMatMulPackedGemvByMWithNR<CudaT, 1, 1, kCompileType>(stream, a, params, y, m);
    return;
  }
  if (params.n >= 512 && params.k >= 2560) {
    LaunchMatMulPackedGemvByMWithNR<CudaT, 2, 2, kCompileType>(stream, a, params, y, m);
    return;
  }
  LaunchMatMulPackedGemvByMWithNR<CudaT, 1, 1, kCompileType>(stream, a, params, y, m);
}

template <int KW, int MT>
void LaunchMatMulPackedBf16TcWarpKSplit(
    cudaStream_t stream,
    const __nv_bfloat16* a,
    const MatMulPackedParams& params,
    __nv_bfloat16* y,
    int m_int) {
  using namespace onnxruntime::contrib::gguf_quant;
  constexpr int kRowsPerBlock = MT * 16;
  const dim3 tc_grid(static_cast<unsigned int>((params.n + kRowsPerBlock - 1) / kRowsPerBlock));
  constexpr unsigned int kBlock = KW * 32;
  switch (params.ggml_type) {
    case kGgmlTypeQ8_0:
      MatMulPackedBf16TcWarpKSplitKernel<KW, MT, kGgmlTypeQ8_0><<<tc_grid, kBlock, 0, stream>>>(a, params, y, m_int);
      return;
    case kGgmlTypeQ6_K:
      MatMulPackedBf16TcWarpKSplitKernel<KW, MT, kGgmlTypeQ6_K><<<tc_grid, kBlock, 0, stream>>>(a, params, y, m_int);
      return;
    case kGgmlTypeQ5_K:
      MatMulPackedBf16TcWarpKSplitKernel<KW, MT, kGgmlTypeQ5_K><<<tc_grid, kBlock, 0, stream>>>(a, params, y, m_int);
      return;
    case kGgmlTypeIQ4_XS:
      MatMulPackedBf16TcWarpKSplitKernel<KW, MT, kGgmlTypeIQ4_XS><<<tc_grid, kBlock, 0, stream>>>(a, params, y, m_int);
      return;
    default:
      MatMulPackedBf16TcWarpKSplitKernel<KW, MT, -1><<<tc_grid, kBlock, 0, stream>>>(a, params, y, m_int);
      return;
  }
}

template <int KW, int MT, int kChunk>
void LaunchMatMulPackedBf16ContiguousK(
    cudaStream_t stream,
    const __nv_bfloat16* a,
    const MatMulPackedParams& params,
    __nv_bfloat16* y,
    int rows) {
  const dim3 grid(static_cast<unsigned int>((params.n + MT * 16 - 1) / (MT * 16)));
  using namespace onnxruntime::contrib::gguf_quant;
  switch (params.ggml_type) {
    case kGgmlTypeQ8_0:
      MatMulPackedBf16TcWarpKSplitKernel<KW, MT, kGgmlTypeQ8_0, kChunk>
          <<<grid, KW * 32, 0, stream>>>(a, params, y, rows);
      return;
    case kGgmlTypeIQ2_XXS:
      MatMulPackedBf16TcWarpKSplitKernel<KW, MT, kGgmlTypeIQ2_XXS, kChunk>
          <<<grid, KW * 32, 0, stream>>>(a, params, y, rows);
      return;
    case kGgmlTypeNVFP4:
      MatMulPackedBf16TcWarpKSplitKernel<KW, MT, kGgmlTypeNVFP4, kChunk>
          <<<grid, KW * 32, 0, stream>>>(a, params, y, rows);
      return;
    default:
      ORT_THROW("MatMulPacked contiguous-K kernel does not support packed_type=", params.ggml_type);
  }
}

template <typename CudaT>
Status DispatchMatMulPackedCuda(
    cublasHandle_t cublas,
    cudaStream_t stream,
    bool is_bf16,
    const CudaT* a,
    const MatMulPackedParams& params,
    void* scratch_w,
    CudaT* y,
    int64_t m) {
  ORT_RETURN_IF_NOT(params.k > 0 && params.k <= kMatMulPackedMaxDimension &&
                        params.n > 0 && params.n <= kMatMulPackedMaxDimension &&
                        m >= 0 && m <= std::numeric_limits<int>::max(),
                    "MatMulPacked dimensions exceed the supported CUDA index range");
  if (m == 0) return Status::OK();
  ORT_RETURN_IF_NOT((params.k % 32) == 0, "MatMulPacked requires K to be a multiple of 32");

  if (m <= 8) {
    using namespace onnxruntime::contrib::gguf_quant;
    const int m_int = static_cast<int>(m);

    if constexpr (std::is_same<CudaT, __nv_bfloat16>::value) {
      if (params.bf16_tc_supported && !params.allow_unrounded_gemv && params.k % 64 == 0 &&
          (params.ggml_type == kGgmlTypeQ8_0 || params.ggml_type == kGgmlTypeIQ2_XXS ||
           params.ggml_type == kGgmlTypeNVFP4)) {
        const bool wide = params.ggml_type == kGgmlTypeNVFP4 && params.k % 128 == 0;
        if (params.n >= 8192) {
          if (wide)
            LaunchMatMulPackedBf16ContiguousK<4, 2, 128>(stream, a, params, y, m_int);
          else
            LaunchMatMulPackedBf16ContiguousK<4, 2, 64>(stream, a, params, y, m_int);
        } else {
          if (wide)
            LaunchMatMulPackedBf16ContiguousK<16, 1, 128>(stream, a, params, y, m_int);
          else
            LaunchMatMulPackedBf16ContiguousK<16, 1, 64>(stream, a, params, y, m_int);
        }
        return CUDA_CALL(cudaGetLastError());
      }
      if ((params.n >= kMatMulPackedBf16TcMinN || !params.allow_unrounded_gemv) &&
          params.bf16_tc_supported) {
        LaunchMatMulPackedBf16TcWarpKSplit<4, 2>(stream, a, params, y, m_int);
        return CUDA_CALL(cudaGetLastError());
      }
    }

    constexpr int kWarpsPerBlock = 8;
    const dim3 grid(static_cast<unsigned int>((params.n + kWarpsPerBlock - 1) / kWarpsPerBlock));

    switch (params.ggml_type) {
      case kGgmlTypeQ8_0:
        LaunchMatMulPackedGemvByM<CudaT, kGgmlTypeQ8_0>(stream, grid, a, params, y, m_int);
        return CUDA_CALL(cudaGetLastError());
      case kGgmlTypeQ6_K:
        LaunchMatMulPackedGemvByM<CudaT, kGgmlTypeQ6_K>(stream, grid, a, params, y, m_int);
        return CUDA_CALL(cudaGetLastError());
      case kGgmlTypeQ5_K:
        LaunchMatMulPackedGemvByM<CudaT, kGgmlTypeQ5_K>(stream, grid, a, params, y, m_int);
        return CUDA_CALL(cudaGetLastError());
      case kGgmlTypeIQ4_XS:
        LaunchMatMulPackedGemvByM<CudaT, kGgmlTypeIQ4_XS>(stream, grid, a, params, y, m_int);
        return CUDA_CALL(cudaGetLastError());
      case kGgmlTypeBF16:
        LaunchMatMulPackedGemvByM<CudaT, kGgmlTypeBF16>(stream, grid, a, params, y, m_int);
        return CUDA_CALL(cudaGetLastError());
      default:
        LaunchMatMulPackedGemvByM<CudaT, -1>(stream, grid, a, params, y, m_int);
        return CUDA_CALL(cudaGetLastError());
    }
  }

  ORT_RETURN_IF_NOT(scratch_w != nullptr, "MatMulPacked prefill requires non-null scratch_w");
  if constexpr (std::is_same_v<CudaT, half>) {
    if (params.fp32_prefill) {
      return LaunchMatMulPackedFp32Prefill(cublas, stream, a, params, static_cast<float*>(scratch_w), y,
                                           static_cast<int>(m));
    }
  }

  // FP32 compute alone still permits FP16/BF16 split-K reductions.
  cublasMath_t previous_math_mode = CUBLAS_DEFAULT_MATH;
  CUBLAS_RETURN_IF_ERROR(cublasGetMathMode(cublas, &previous_math_mode));
  const auto math_mode = static_cast<cublasMath_t>(
      previous_math_mode | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  if (math_mode != previous_math_mode) {
    CUBLAS_RETURN_IF_ERROR(cublasSetMathMode(cublas, math_mode));
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cudaDataType_t data_type = is_bf16 ? CUDA_R_16BF : CUDA_R_16F;
  Status gemm_status = Status::OK();
  for (int n0 = 0; n0 < params.n;) {
    MatMulPackedParams tile_params = params;
    tile_params.n = std::min(kMatMulPackedPrefillNTile, params.n - n0);
    tile_params.b_data = params.b_data + static_cast<size_t>(SafeInt<size_t>(n0) * params.row_bytes);
    const dim3 grid(static_cast<unsigned int>((tile_params.n + 7) / 8));
    MatMulPackedDequantKernel<CudaT><<<grid, 256, 0, stream>>>(tile_params, static_cast<CudaT*>(scratch_w));
    gemm_status = CUDA_CALL(cudaGetLastError());
    if (!gemm_status.IsOK()) break;

    if (params.global_scale) {
      auto* output = reinterpret_cast<float*>(
          static_cast<CudaT*>(scratch_w) + static_cast<size_t>(SafeInt<size_t>(tile_params.n) * params.k));
      for (int m0 = 0; m0 < m;) {
        const int tile_m = static_cast<int>(std::min<int64_t>(kMatMulPackedPrefillMTile, m - m0));
        gemm_status = CUBLAS_CALL(cublasGemmEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, tile_params.n, tile_m, params.k,
            &alpha, scratch_w, data_type, params.k, a + static_cast<int64_t>(m0) * params.k,
            data_type, params.k, &beta, output, CUDA_R_32F, tile_params.n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        if (!gemm_status.IsOK()) break;
        const dim3 output_grid((static_cast<unsigned int>(tile_params.n) + 255) / 256,
                               static_cast<unsigned int>(tile_m));
        MatMulPackedCastTileKernel<<<output_grid, 256, 0, stream>>>(
            output, y + static_cast<int64_t>(m0) * params.n + n0,
            tile_params.n, tile_params.n, params.n, params.global_scale);
        gemm_status = CUDA_CALL(cudaGetLastError());
        if (!gemm_status.IsOK()) break;
        m0 += tile_m;
      }
      if (!gemm_status.IsOK()) break;
      n0 += tile_params.n;
      continue;
    }

    // Column-major [N, M] is row-major [M, N]; retain the full N leading dimension for every tile.
    gemm_status = CUBLAS_CALL(cublasGemmEx(
        cublas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        tile_params.n,
        static_cast<int>(m),
        params.k,
        &alpha,
        scratch_w,
        data_type,
        params.k,
        a,
        data_type,
        params.k,
        &beta,
        y + n0,
        data_type,
        params.n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if (!gemm_status.IsOK()) break;
    n0 += tile_params.n;
  }
  if (math_mode != previous_math_mode) {
    CUBLAS_RETURN_IF_ERROR(cublasSetMathMode(cublas, previous_math_mode));
  }
  return gemm_status;
}

}  // namespace

Status GetMatMulPackedWorkspaceBytes(const MatMulPackedParams& params, int64_t m, size_t& workspace_bytes) {
  ORT_RETURN_IF_NOT(params.k > 0 && params.k <= kMatMulPackedMaxDimension &&
                        params.n > 0 && params.n <= kMatMulPackedMaxDimension &&
                        m >= 0 && m <= std::numeric_limits<int>::max(),
                    "MatMulPacked workspace parameters exceed the supported range");
  if (m <= 8) {
    workspace_bytes = 0;
    return Status::OK();
  }
  const size_t tile_n = std::min(kMatMulPackedPrefillNTile, params.n);
  const size_t tile_m = std::min<int64_t>(kMatMulPackedPrefillMTile, m);
  SafeInt<size_t> count = SafeInt<size_t>(tile_n) * params.k;
  if (params.fp32_prefill) {
    count += SafeInt<size_t>(tile_m) * params.k;
    count += SafeInt<size_t>(tile_m) * tile_n;
  }
  const size_t element_bytes = params.fp32_prefill ? sizeof(float) : sizeof(uint16_t);
  if (params.global_scale && !params.fp32_prefill) {
    count += SafeInt<size_t>(tile_m) * tile_n * (sizeof(float) / sizeof(uint16_t));
  }
  ORT_RETURN_IF_NOT(IAllocator::CalcMemSizeForArray(count, element_bytes, &workspace_bytes) &&
                        workspace_bytes <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()),
                    "MatMulPacked workspace byte size exceeds the supported address range");
  return Status::OK();
}

Status GetMatMulPackedBf16KernelSupport(int device_id, bool& supported) {
  int previous_device = -1;
  CUDA_RETURN_IF_ERROR(cudaGetDevice(&previous_device));
  auto restore_device = gsl::finally([previous_device, device_id]() {
    if (previous_device != device_id) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaSetDevice(previous_device)));
    }
  });
  if (previous_device != device_id) {
    CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
  }
  cudaFuncAttributes attributes{};
  // All BF16 MMA instantiations in this translation unit use the same virtual architecture.
  CUDA_RETURN_IF_ERROR(cudaFuncGetAttributes(&attributes, MatMulPackedBf16TcWarpKSplitKernel<4, 2, -1>));
  supported = attributes.ptxVersion >= 80;
  return Status::OK();
}

Status LaunchMatMulPackedCuda(
    cublasHandle_t cublas,
    cudaStream_t stream,
    bool is_bf16,
    const void* a,
    const MatMulPackedParams& params,
    void* scratch_w,
    void* y,
    int64_t m) {
  if (is_bf16) {
    return DispatchMatMulPackedCuda<__nv_bfloat16>(
        cublas, stream, true, static_cast<const __nv_bfloat16*>(a), params,
        scratch_w, static_cast<__nv_bfloat16*>(y), m);
  }
  return DispatchMatMulPackedCuda<half>(
      cublas, stream, false, static_cast<const half*>(a), params,
      scratch_w, static_cast<half*>(y), m);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
