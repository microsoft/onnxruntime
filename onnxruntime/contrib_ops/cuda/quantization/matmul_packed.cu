// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_packed.h"
#include "contrib_ops/cuda/quantization/gguf_quant_cuda.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Warp-K-Split Tensor-Core BF16 GEMV/GEMM kernel for M in 1..8 on large-N projections (e.g. lm_head).
// KW warps (KW*32 threads) split K across the KW warps with zero __syncthreads() in the K-loop,
// while computing MT*16 output columns per CTA.
template <int KW, int MT, int kCompileType>
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
  const __nv_bfloat16* a_row = a + static_cast<int64_t>(a_valid ? g : 0) * p.k;

  const int num_sb32 = p.k >> 5;
  for (int sb32_idx = warp; sb32_idx < num_sb32; sb32_idx += KW) {
    const uint4 b_vec = a_valid
                            ? __ldg(reinterpret_cast<const uint4*>(a_row + sb32_idx * 32 + t * 8))
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
      // mma.sync requires every lane to participate, including partial output tiles.
      const uint4 e_vec = gguf_cuda::PackFloat8ToBf16x2(w_lo);
      const uint4 o_vec = gguf_cuda::PackFloat8ToBf16x2(w_hi);
      gguf_cuda::MmaBf16(acc[m], e_vec.x, o_vec.x, e_vec.y, o_vec.y, b_vec.x, b_vec.y);
      gguf_cuda::MmaBf16(acc[m], e_vec.z, o_vec.z, e_vec.w, o_vec.w, b_vec.z, b_vec.w);
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
      y[static_cast<int64_t>(row_m) * p.n + col] = __float2bfloat16_rn(sum);
    }
  }
}

template <typename CudaT, int M_FIXED, int kCompileType>
__global__ __launch_bounds__(256) void MatMulPackedGemvKernel(
    const CudaT* __restrict__ a,
    MatMulPackedParams p,
    CudaT* __restrict__ y,
    int m_rt) {
  constexpr int kWarpsPerBlock = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n_out = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
  if (n_out >= p.n) return;

  const int M = M_FIXED > 0 ? M_FIXED : m_rt;
  const uint8_t* row_ptr = p.b_data + static_cast<int64_t>(n_out) * p.row_bytes;

  float acc[M_FIXED > 0 ? M_FIXED : 8];
#pragma unroll
  for (int m = 0; m < (M_FIXED > 0 ? M_FIXED : 8); ++m) {
    acc[m] = 0.0f;
  }

  const int total_quarters = p.k >> 3;
#pragma unroll 1
  for (int q_idx = lane; q_idx < total_quarters; q_idx += 32) {
    const int sb32_idx = q_idx >> 2;
    const int l = q_idx & 3;
    float w_vals[8];
    gguf_cuda::DecodeGgufSubBlock8Tpl<kCompileType>(row_ptr, p.ggml_type, sb32_idx, l, w_vals);
    const int k_offset = sb32_idx * 32 + (l << 3);

    if constexpr (M_FIXED > 0) {
#pragma unroll
      for (int m = 0; m < M_FIXED; ++m) {
        const CudaT* a_ptr = a + static_cast<int64_t>(m) * p.k + k_offset;
        const uint4 v0 = __ldg(reinterpret_cast<const uint4*>(a_ptr));
        const CudaT* a0 = reinterpret_cast<const CudaT*>(&v0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          acc[m] += w_vals[j] * gguf_cuda::CudaTypeToFloat<CudaT>(a0[j]);
        }
      }
    } else {
      for (int m = 0; m < M; ++m) {
        const CudaT* a_ptr = a + static_cast<int64_t>(m) * p.k + k_offset;
        const uint4 v0 = __ldg(reinterpret_cast<const uint4*>(a_ptr));
        const CudaT* a0 = reinterpret_cast<const CudaT*>(&v0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          acc[m] += w_vals[j] * gguf_cuda::CudaTypeToFloat<CudaT>(a0[j]);
        }
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int m = 0; m < (M_FIXED > 0 ? M_FIXED : 8); ++m) {
      acc[m] += __shfl_down_sync(0xFFFFFFFFu, acc[m], offset);
    }
  }

  if (lane == 0) {
    if constexpr (M_FIXED > 0) {
#pragma unroll
      for (int m = 0; m < M_FIXED; ++m) {
        y[static_cast<int64_t>(m) * p.n + n_out] = gguf_cuda::FloatToCudaType<CudaT>(acc[m]);
      }
    } else {
      for (int m = 0; m < M; ++m) {
        y[static_cast<int64_t>(m) * p.n + n_out] = gguf_cuda::FloatToCudaType<CudaT>(acc[m]);
      }
    }
  }
}

template <typename CudaT>
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
    if constexpr (std::is_same<CudaT, __nv_bfloat16>::value) {
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

template <typename CudaT, int kCompileType>
void LaunchMatMulPackedGemvByM(
    cudaStream_t stream,
    dim3 grid,
    const CudaT* a,
    const MatMulPackedParams& params,
    CudaT* y,
    int m) {
  const dim3 block(256);
  switch (m) {
    case 1:
      MatMulPackedGemvKernel<CudaT, 1, kCompileType><<<grid, block, 0, stream>>>(a, params, y, 1);
      return;
    case 2:
      MatMulPackedGemvKernel<CudaT, 2, kCompileType><<<grid, block, 0, stream>>>(a, params, y, 2);
      return;
    case 4:
      MatMulPackedGemvKernel<CudaT, 4, kCompileType><<<grid, block, 0, stream>>>(a, params, y, 4);
      return;
    case 7:
      MatMulPackedGemvKernel<CudaT, 7, kCompileType><<<grid, block, 0, stream>>>(a, params, y, 7);
      return;
    case 8:
      MatMulPackedGemvKernel<CudaT, 8, kCompileType><<<grid, block, 0, stream>>>(a, params, y, 8);
      return;
    default:
      MatMulPackedGemvKernel<CudaT, 0, kCompileType><<<grid, block, 0, stream>>>(a, params, y, m);
      return;
  }
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

template <typename CudaT>
Status DispatchMatMulPackedCuda(
    cublasHandle_t cublas,
    cudaStream_t stream,
    bool is_bf16,
    const CudaT* a,
    const MatMulPackedParams& params,
    CudaT* scratch_w,
    CudaT* y,
    int64_t m) {
  if (m == 0) return Status::OK();
  ORT_RETURN_IF_NOT((params.k % 32) == 0, "MatMulPacked requires K to be a multiple of 32");

  if (m <= 8) {
    using namespace onnxruntime::contrib::gguf_quant;
    const int m_int = static_cast<int>(m);

    if constexpr (std::is_same<CudaT, __nv_bfloat16>::value) {
      if (params.n >= 65536) {
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
      default:
        LaunchMatMulPackedGemvByM<CudaT, -1>(stream, grid, a, params, y, m_int);
        return CUDA_CALL(cudaGetLastError());
    }
  }

  ORT_RETURN_IF_NOT(scratch_w != nullptr, "MatMulPacked prefill requires non-null scratch_w");
  constexpr int kWarpsPerBlock = 8;
  const dim3 grid(static_cast<unsigned int>((params.n + kWarpsPerBlock - 1) / kWarpsPerBlock));
  const dim3 block(256);
  MatMulPackedDequantKernel<CudaT><<<grid, block, 0, stream>>>(params, scratch_w);
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cudaDataType_t data_type = is_bf16 ? CUDA_R_16BF : CUDA_R_16F;
  return CUBLAS_CALL(cublasGemmEx(
      cublas,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      params.n,
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
      y,
      data_type,
      params.n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

}  // namespace

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
        static_cast<__nv_bfloat16*>(scratch_w), static_cast<__nv_bfloat16*>(y), m);
  }
  return DispatchMatMulPackedCuda<half>(
      cublas, stream, false, static_cast<const half*>(a), params,
      static_cast<half*>(scratch_w), static_cast<half*>(y), m);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
