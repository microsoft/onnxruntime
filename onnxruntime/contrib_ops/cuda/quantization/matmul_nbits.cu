// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "matmul_nbits.cuh"

#if !defined(USE_MIGRAPHX) && !defined(USE_ROCM)
#include "blk_q4/f16_gemm_sm80.h"
#include "gemm/device/quant_b4_gemm.h"
#endif  // !defined(USE_MIGRAPHX) && !defined(USE_ROCM)

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ __forceinline__ T WarpUniform(T value) {
  struct {
    union {
      T value;
      uint32_t asInt;
    };
  } p;
  p.value = value;
  p.asInt = WARP_SHFL((unsigned)p.asInt, 0);
  return p.value;
}

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
// Convert 8 4bits integer stored in one uint32_t to 8 halfs.
// 8 4bits with order 0,1,2,3,4,5,6,7,8 will be converted to 8 halfs with order 0,4,1,5,2,6,3,7
__device__ __forceinline__ void Convert8xInt4To8xHalfs(uint32_t value, half2* half_2x4) {
  uint32_t* h = reinterpret_cast<uint32_t*>(half_2x4);

  // From https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
  // First, we extract the i4s and construct an intermediate fp16 number.
  constexpr uint32_t kImmLut = (0xf0 & 0xcc) | 0xaa;
  constexpr uint32_t kBottomMask = 0x000f000f;
  constexpr uint32_t kTopMask = 0x00f000f0;
  constexpr uint32_t kI4sToF16sMagicNum = 0x64006400;

  // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
  // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
  // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
  // elt_67 to fp16 without having to shift them to the bottom bits before hand.

  // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
  // immediately before required.
  const uint32_t top_i4s = value >> 8;
  // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(value), "n"(kBottomMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));
  // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(value), "n"(kTopMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));
  // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(kBottomMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));
  // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(kTopMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));

  // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
  // half2 ctor. In this case, I chose performance reliability over code readability.

  // This is the half2 {1024, 1024} represented as an integer.
  constexpr uint32_t kFp16TopMagicNum = 0x64006400;
  // This is the half2 {1 / 16, 1 / 16} represented as an integer.
  constexpr uint32_t kOneSixteenth = 0x2c002c00;
  // This is the half2 {-64, -64} represented as an integer.
  constexpr uint32_t kNeg64 = 0xd400d400;

  // Finally, we construct the output numbers.
  // Convert elt_01
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(kFp16TopMagicNum));
  // Convert elt_23
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(kOneSixteenth), "r"(kNeg64));
  // Convert elt_45
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(kFp16TopMagicNum));
  // Convert elt_67
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(kOneSixteenth), "r"(kNeg64));
}

__device__ __forceinline__ void AccumulateEightElements(uint32_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  constexpr uint32_t kLowHalf2 = 0x5410;
  constexpr uint32_t kHighHalf2 = 0x7632;

  uint4 vec_permuted;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.x) : "r"(vec_a.x), "r"(vec_a.z), "r"(kLowHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.y) : "r"(vec_a.x), "r"(vec_a.z), "r"(kHighHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.z) : "r"(vec_a.y), "r"(vec_a.w), "r"(kLowHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.w) : "r"(vec_a.y), "r"(vec_a.w), "r"(kHighHalf2));

  half2 elements[4];  // [04, 15, 26, 37]

  Convert8xInt4To8xHalfs(values_quant, elements);

  half2 v0 = elements[0] * scale_half2 + zp_adjust2;
  half2 v1 = elements[1] * scale_half2 + zp_adjust2;
  half2 v2 = elements[2] * scale_half2 + zp_adjust2;
  half2 v3 = elements[3] * scale_half2 + zp_adjust2;

  half2* sums_half2 = reinterpret_cast<half2*>(sums);
  sums_half2[0] = sums_half2[0] + v0 * (*(reinterpret_cast<half2*>(&(vec_permuted.x))));
  sums_half2[1] = sums_half2[1] + v1 * (*(reinterpret_cast<half2*>(&(vec_permuted.y))));
  sums_half2[2] = sums_half2[2] + v2 * (*(reinterpret_cast<half2*>(&(vec_permuted.z))));
  sums_half2[3] = sums_half2[3] + v3 * (*(reinterpret_cast<half2*>(&(vec_permuted.w))));
}
#else
__device__ __forceinline__ void AccumulateEightElements(uint32_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  half2 element01 = __halves2half2(__uint2half_rn(values_quant & 0xF), __uint2half_rn((values_quant >> 4) & 0xF));
  half2 element23 = __halves2half2(__uint2half_rn((values_quant >> 8) & 0xF), __uint2half_rn((values_quant >> 12) & 0xF));
  half2 element45 = __halves2half2(__uint2half_rn((values_quant >> 16) & 0xF), __uint2half_rn((values_quant >> 20) & 0xF));
  half2 element67 = __halves2half2(__uint2half_rn((values_quant >> 24) & 0xF), __uint2half_rn((values_quant >> 28) & 0xF));

  half2 v0 = element01 * scale_half2 + zp_adjust2;
  half2 v1 = element23 * scale_half2 + zp_adjust2;
  half2 v2 = element45 * scale_half2 + zp_adjust2;
  half2 v3 = element67 * scale_half2 + zp_adjust2;

  half2* sums_half2 = reinterpret_cast<half2*>(sums);
  sums_half2[0] = sums_half2[0] + v0 * (*(reinterpret_cast<half2*>(&(vec_a.x))));
  sums_half2[1] = sums_half2[1] + v1 * (*(reinterpret_cast<half2*>(&(vec_a.y))));
  sums_half2[2] = sums_half2[2] + v2 * (*(reinterpret_cast<half2*>(&(vec_a.z))));
  sums_half2[3] = sums_half2[3] + v3 * (*(reinterpret_cast<half2*>(&(vec_a.w))));
}
#endif

__device__ __forceinline__ void AccumulateEightElements(uint32_t values_quant, float scale, uint8_t zp, const float* a, float* sums) {
  float4 a_vec_0 = *(reinterpret_cast<const float4*>(a));
  float4 a_vec_1 = *(reinterpret_cast<const float4*>(a + 4));

  float zp_adjust = -scale * zp;
  float v0 = float(values_quant & 0xF) * scale + zp_adjust;
  float v1 = float((values_quant >> 4) & 0xF) * scale + zp_adjust;
  float v2 = float((values_quant >> 8) & 0xF) * scale + zp_adjust;
  float v3 = float((values_quant >> 12) & 0xF) * scale + zp_adjust;
  float v4 = float((values_quant >> 16) & 0xF) * scale + zp_adjust;
  float v5 = float((values_quant >> 20) & 0xF) * scale + zp_adjust;
  float v6 = float((values_quant >> 24) & 0xF) * scale + zp_adjust;
  float v7 = float((values_quant >> 28) & 0xF) * scale + zp_adjust;

  sums[0] += v0 * a_vec_0.x;
  sums[1] += v1 * a_vec_0.y;
  sums[2] += v2 * a_vec_0.z;
  sums[3] += v3 * a_vec_0.w;
  sums[4] += v4 * a_vec_1.x;
  sums[5] += v5 * a_vec_1.y;
  sums[6] += v6 * a_vec_1.z;
  sums[7] += v7 * a_vec_1.w;
}

constexpr int kColsPerThreadBlock = 8;
constexpr int kElementsPerThreadPerIteration = 8;
constexpr int kWarpSize = GPU_WARP_SIZE;

// kernel for 4bits quantized gemv, i.e., computing A(1,K) x B(K, N)
// B(K, N) is quantized blockwise with 4bits and stored as [N, (K + block_size - 1)/block_size, blob]
// The thread block size is (kWarpSize, kColsPerThreadBlock) and grid size is (N/kColsPerThreadBlock, 1)
// Each thread block computes [1, K] x [kColsPerThreadBlock, (K + block_size - 1)/block_size, blob],
//     i.e., computing kColsPerThreadBlock per block and a warp reduce (1, K) x (K)
template <class T, int block_size, bool has_zero_point>
__global__ void __launch_bounds__(kWarpSize * kColsPerThreadBlock) MatMulFloatInt4Kernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  const int n_block_id = blockIdx.x;
  const int m_id = blockIdx.y;
  const int lane_id = threadIdx.x;
  const int warp_id = WarpUniform(threadIdx.y);
  const int n_id = n_block_id * kColsPerThreadBlock + warp_id;
  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;

  extern __shared__ char shared_buffer[];
  // load scale to shared buffer
  T* b_scale_vec = (T*)shared_buffer;
  int offset = n_block_id * kColsPerThreadBlock * blocks_per_K;
  for (int i = warp_id * kWarpSize + lane_id; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
    b_scale_vec[i] = scales_data[offset + i];
  }

  uint8_t* b_zp_vec;
  (void)b_zp_vec;
  if constexpr (has_zero_point) {
    b_zp_vec = reinterpret_cast<uint8_t*>(b_scale_vec + kColsPerThreadBlock * blocks_per_K);
    const int b_zp_k = (blocks_per_K + 1) / 2;
    int zp_offset = n_block_id * kColsPerThreadBlock * b_zp_k;
    for (int i = warp_id * kWarpSize + lane_id; i < kColsPerThreadBlock * b_zp_k; i += kColsPerThreadBlock * kWarpSize) {
      b_zp_vec[2 * i] = (zero_points[zp_offset + i] & 0x0f);
      b_zp_vec[2 * i + 1] = (zero_points[zp_offset + i] >> 4);
    }
    b_zp_vec += warp_id * b_zp_k * 2;
  }
  __syncthreads();

  a_data += m_id * k + (lane_id << 3);

  b_scale_vec += warp_id * blocks_per_K;

  T sums[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int k_id = 0;
  int t_meta_k = lane_id * 8 / block_size;
  b_data_quant += n_id * blocks_per_K * (block_size / 2) + lane_id * 4;

#define UnRollReduction(unroll_size)                                                              \
  do {                                                                                            \
    constexpr int kUnroll = unroll_size;                                                          \
    constexpr int kUnrollMask = 0xffffffff & (~(kUnroll * k_per_iter - 1));                       \
    for (; k_id < (k & kUnrollMask); k_id += kUnroll * k_per_iter) {                              \
      _Pragma("unroll") for (int i = 0; i < kUnroll; i++) {                                       \
        uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_per_iter / 2 * i)); \
        T scale = b_scale_vec[t_meta_k + k_per_iter / block_size * i];                            \
        uint8_t zp = 8;                                                                           \
        if constexpr (has_zero_point) {                                                           \
          zp = b_zp_vec[t_meta_k + k_per_iter / block_size * i];                                  \
        }                                                                                         \
        AccumulateEightElements(value, scale, zp, a_data + k_id + i * k_per_iter, sums);          \
      }                                                                                           \
      b_data_quant += k_per_iter / 2 * kUnroll;                                                   \
      t_meta_k += k_per_iter / block_size * kUnroll;                                              \
    }                                                                                             \
  } while (false)

  UnRollReduction(16);
  UnRollReduction(4);
  UnRollReduction(1);
#undef UnRollReduction

  // handle reminder
  if (k_id + lane_id * 8 < k) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant));
    T scale = b_scale_vec[t_meta_k];
    uint8_t zp = 8;
    if constexpr (has_zero_point) {
      zp = b_zp_vec[t_meta_k];
    }
    AccumulateEightElements(value, scale, zp, a_data + k_id, sums);
  }

  float sum = (float)(sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7]);
  // warp reduction
  for (int i = kWarpSize / 2; i > 0; i = i / 2) {
    sum += WARP_SHFL_DOWN(sum, i);
  }

  if (lane_id == 0) {
    output[m_id * n + n_id] = sum;
  }
}  // namespace cuda

template <class T>
bool TryMatMul4Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream) {
  if (n % kColsPerThreadBlock != 0 || k % 8 != 0 || m > 1) {
    return false;
  }
  dim3 blocks((n + kColsPerThreadBlock - 1) / kColsPerThreadBlock, m);
  dim3 threads(kWarpSize, kColsPerThreadBlock);
  int blocks_per_K = (k + block_size - 1) / block_size;
  int shared_mem_size = sizeof(T) * blocks_per_K * kColsPerThreadBlock +
                        (zero_points != nullptr ? (blocks_per_K + 1) / 2 * kColsPerThreadBlock * 2 : 0);
  if (shared_mem_size > shared_mem_per_block) {
    return false;
  }

#define MatMulFloatInt4KernelDispatch(block_size)                                              \
  if (nullptr != zero_points) {                                                                \
    MatMulFloatInt4Kernel<T, block_size, true><<<blocks, threads, shared_mem_size, stream>>>(  \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);        \
  } else {                                                                                     \
    MatMulFloatInt4Kernel<T, block_size, false><<<blocks, threads, shared_mem_size, stream>>>( \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);        \
  }

  if (16 == block_size) {
    MatMulFloatInt4KernelDispatch(16);
  } else if (32 == block_size) {
    MatMulFloatInt4KernelDispatch(32);
  } else if (64 == block_size) {
    MatMulFloatInt4KernelDispatch(64);
  } else if (128 == block_size) {
    MatMulFloatInt4KernelDispatch(128);
  } else {
    ORT_THROW("block size ", block_size, " is not supported");
  }

#undef MatMulFloatInt4KernelDispatch

  return true;
}

template bool TryMatMul4Bits<float>(
    float* output,
    const float* a_data,
    const uint8_t* b_data_quant,
    const float* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);

template bool TryMatMul4Bits<half>(
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);


#if !defined(USE_MIGRAPHX) && !defined(USE_ROCM)

/**
 * @brief Helper function to run the GEMM kernel for 4bits quantized gemm on SM80.
 * Only support fp16 for now.
*/
template<
    typename ElementT,
    int block_size,
    bool column_wise_blocking,
    bool small_m,
    bool has_offsets>
Status blkq4_gemm_sm80(int m, int n, int k, cudaStream_t stream,
                     gsl::span<ElementT const> a,
                     gsl::span<uint8_t const> weights,
                     gsl::span<ElementT const> scales,
                     gsl::span<uint8_t const> offsets,
                     gsl::span<ElementT> output) {
  static_assert(std::is_same<ElementT, half>::value
                || std::is_same<ElementT, MLFloat16>::value
                || std::is_same<ElementT, cutlass::half_t>::value,
                "Only support fp16 for now");
  using ElementDequant = cutlass::half_t;
  using QuantBlocking =
    typename std::conditional<column_wise_blocking,
                     cutlass::MatrixShape<block_size, 1>,
                     cutlass::MatrixShape<1, block_size>>::type;

  using GemmRunner = onnxruntime::cuda::BlkQ4F16GemmImpl<ElementDequant, QuantBlocking, small_m, has_offsets>;

  using ElementAccumulator = typename GemmRunner::ElementAccumulator;
  using ElementComputeEpilogue = typename GemmRunner::ElementComputeEpilogue;
  using ElementOutput = typename GemmRunner::ElementOutput;
  using ElementW = typename GemmRunner::ElementW;
  using ElementWPack = typename GemmRunner::ElementWPack;
  using ElementQScale = typename GemmRunner::ElementQScale;
  using ElementQOffset = typename GemmRunner::ElementQOffset;

  using LayoutInputA = typename GemmRunner::LayoutInputA;
  using LayoutOutput = typename GemmRunner::LayoutOutput;
  using LayoutInputWPack = typename GemmRunner::LayoutInputWPack;
  using LayoutInputQScale = typename GemmRunner::LayoutInputQScale;

  const cutlass::gemm::GemmCoord problem_size = {m, n, k};

  ORT_RETURN_IF_NOT(a.size_bytes() == m * k * sizeof(ElementDequant), "Activation tensor size is not correct: ", a.size_bytes(), " vs m: ", m, "k: ", k , " size: ", m * k * sizeof(ElementDequant));
  cutlass::TensorRef<ElementDequant const, LayoutInputA> ref_a(
    reinterpret_cast<ElementDequant const *>(a.data()),
    LayoutInputA::packed({m, k}));

  ORT_RETURN_IF_NOT(weights.size_bytes() == k/2 * n/2 * sizeof(ElementWPack), "weights size is not correct");
  cutlass::TensorRef<ElementWPack const, LayoutInputWPack> ref_W(
    reinterpret_cast<ElementWPack const *>(weights.data()),
    LayoutInputWPack::packed({k/2, n/2}));

  ORT_RETURN_IF_NOT(scales.size_bytes() == (k/QuantBlocking::kRow) * (n/QuantBlocking::kColumn) * sizeof(ElementQScale),
              "scales size is not correct");
  cutlass::TensorRef<ElementQScale const, LayoutInputQScale> ref_scales(
    reinterpret_cast<ElementQScale const *>(scales.data()),
    LayoutInputQScale::packed({k/QuantBlocking::kRow, n/QuantBlocking::kColumn}));

  ORT_RETURN_IF_NOT(output.size_bytes() == m * n * sizeof(ElementOutput), "output size is not correct");
  cutlass::TensorRef<ElementOutput, LayoutOutput> ref_output(
    reinterpret_cast<ElementOutput *>(output.data()),
    LayoutOutput::packed({m, n}));

  // run GEMM
  cutlass::Status status;
  if constexpr (has_offsets) {
    ORT_RETURN_IF_NOT(offsets.size_bytes() == (k/QuantBlocking::kRow) * (n/QuantBlocking::kColumn) * sizeof(ElementQOffset),
                "offsets size is not correct");
    cutlass::TensorRef<ElementQOffset const, LayoutInputQScale> ref_offsets(
      reinterpret_cast<ElementQOffset const *>(offsets.data()),
      LayoutInputQScale::packed({k/QuantBlocking::kRow, n/QuantBlocking::kColumn}));
    status = GemmRunner::run(
      stream, problem_size, ref_a, ref_W, ref_scales, ref_offsets,
      ref_output, ref_output);
  } else {
    status = GemmRunner::run(
      stream, problem_size, ref_a, ref_W, ref_scales,
      ref_output, ref_output);
  }
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess, "Kernel execution failed: ", cutlassGetStatusString(status));
  return Status::OK();
}

/**
 * @brief The GEMM kernel for 4bits quantized gemm on SM80 -- small size gemm version.
 */
template <
    typename ElementT,
    int block_size,
    bool column_wise_blocking,
    bool has_offsets>
Status blkq4_small_gemm_sm80(
    int m, int n, int k, cudaStream_t stream,
    const ElementT* ptr_a,
    size_t lda,
    const uint8_t* ptr_packed_b,
    const ElementT* ptr_scales,
    const uint8_t* ptr_offsets,
    ElementT* ptr_c,
    size_t ldc) {
  using QuantBlocking =
      typename std::conditional<column_wise_blocking,
                                cutlass::MatrixShape<block_size, 1>,
                                cutlass::MatrixShape<1, block_size>>::type;
  using LayoutQmeta =
      typename std::conditional<column_wise_blocking,
                                cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 64>;

  const cutlass::gemm::GemmCoord problem_size = {m, n, k};
  const auto meta_shape = cutlass::make_Coord(problem_size.k() / QuantBlocking::kRow,
                                              problem_size.n() / QuantBlocking::kColumn);
  if ((problem_size.k() % QuantBlocking::kRow != 0) ||
    (problem_size.n() % QuantBlocking::kColumn) != 0){
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Partial quantization block in B not supported!");
  }

  // run GEMM
  size_t meta_stride = static_cast<size_t>(LayoutQmeta::packed(meta_shape).stride(0));
  const void* ptr_zp = has_offsets ? ptr_offsets : nullptr;
  size_t zp_byte_stride = has_offsets ? meta_stride * sizeof(uint8_t) : size_t(0);

  cutlass::Status status;

  if (k <= 384) {
    status = mickey::gemm::device::QuantB4Gemm<QuantBlocking, has_offsets, WarpShape, 1, 3>::run(
        stream, problem_size,
        ptr_c, ldc * sizeof(ElementT),
        ptr_a, lda * sizeof(ElementT),
        ptr_packed_b, k * sizeof(uint8_t),
        ptr_scales, meta_stride * sizeof(ElementT),
        ptr_zp, zp_byte_stride);
  } else if (k <= 768) {
    status = mickey::gemm::device::QuantB4Gemm<QuantBlocking, has_offsets, WarpShape, 2, 3>::run(
        stream, problem_size,
        ptr_c, ldc * sizeof(ElementT),
        ptr_a, lda * sizeof(ElementT),
        ptr_packed_b, k * sizeof(uint8_t),
        ptr_scales, meta_stride * sizeof(ElementT),
        ptr_zp, zp_byte_stride);
  } else if (k < 1536) {
    status = mickey::gemm::device::QuantB4Gemm<QuantBlocking, has_offsets, WarpShape, 4, 3>::run(
        stream, problem_size,
        ptr_c, ldc * sizeof(ElementT),
        ptr_a, lda * sizeof(ElementT),
        ptr_packed_b, k * sizeof(uint8_t),
        ptr_scales, meta_stride * sizeof(ElementT),
        ptr_zp, zp_byte_stride);
  } else {
    status = mickey::gemm::device::QuantB4Gemm<QuantBlocking, has_offsets, WarpShape, 8, 3>::run(
        stream, problem_size,
        ptr_c, ldc * sizeof(ElementT),
        ptr_a, lda * sizeof(ElementT),
        ptr_packed_b, k * sizeof(uint8_t),
        ptr_scales, meta_stride * sizeof(ElementT),
        ptr_zp, zp_byte_stride);
  }

  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess, "Kernel execution failed: ", cutlassGetStatusString(status));
  return Status::OK();
}


template<typename ElementT>
Status
blkq4_fp16_gemm_sm80_dispatch(
  int block_size, bool column_wise_blocking, int m, int n, int k, cudaStream_t stream,
  ElementT const* a_ptr, size_t a_size,
  uint8_t const* weights_ptr, size_t weights_size,
  ElementT const* scales_ptr, size_t scales_size,
  uint8_t const* offsets_ptr, size_t offsets_size,
  ElementT* output_ptr, size_t output_size) {
  auto a = gsl::make_span(a_ptr, a_size);
  auto weights = gsl::make_span(weights_ptr, weights_size);
  auto scales = gsl::make_span(scales_ptr, scales_size);
  auto offsets = gsl::make_span(offsets_ptr, offsets_size);
  auto output = gsl::make_span(output_ptr, output_size);

  switch (block_size)
  {
  case 16:
    if (column_wise_blocking) {
      if (m <= 64 && n < 16384) {
        if (offsets.empty())
          return blkq4_small_gemm_sm80<ElementT, 16, true, false>(m, n, k, stream, a_ptr, k, weights_ptr, scales_ptr, offsets_ptr, output_ptr, n);
        else
          return blkq4_small_gemm_sm80<ElementT, 16, true, true>(m, n, k, stream, a_ptr, k, weights_ptr, scales_ptr, offsets_ptr, output_ptr, n);
      }
      if (m > 32) {
        if (offsets.empty())
          return blkq4_gemm_sm80<ElementT, 16, true, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<ElementT, 16, true, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<ElementT, 16, true, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<ElementT, 16, true, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    } else {
      if (m <= 64 && n < 16384) {
        if (offsets.empty())
          return blkq4_small_gemm_sm80<ElementT, 16, false, false>(m, n, k, stream, a_ptr, k, weights_ptr, scales_ptr, offsets_ptr, output_ptr, n);
        else
          return blkq4_small_gemm_sm80<ElementT, 16, false, true>(m, n, k, stream, a_ptr, k, weights_ptr, scales_ptr, offsets_ptr, output_ptr, n);
      }
      if (m > 32) {
        if (offsets.empty())
          return blkq4_gemm_sm80<ElementT, 16, false, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<ElementT, 16, false, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<ElementT, 16, false, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<ElementT, 16, false, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    }
    break;

  case 32:
    if (column_wise_blocking) {
      if (m <= 64 && n < 16384) {
        if (offsets.empty())
          return blkq4_small_gemm_sm80<ElementT, 32, true, false>(m, n, k, stream, a_ptr, k, weights_ptr, scales_ptr, offsets_ptr, output_ptr, n);
        else
          return blkq4_small_gemm_sm80<ElementT, 32, true, true>(m, n, k, stream, a_ptr, k, weights_ptr, scales_ptr, offsets_ptr, output_ptr, n);
      }
      if (m > 32) {
        if (offsets.empty())
          return blkq4_gemm_sm80<ElementT, 32, true, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<ElementT, 32, true, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<ElementT, 32, true, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<ElementT, 32, true, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    } else {
      if (m <= 64 && n < 16384) {
        if (offsets.empty())
          return blkq4_small_gemm_sm80<ElementT, 32, false, false>(m, n, k, stream, a_ptr, k, weights_ptr, scales_ptr, offsets_ptr, output_ptr, n);
        else
          return blkq4_small_gemm_sm80<ElementT, 32, false, true>(m, n, k, stream, a_ptr, k, weights_ptr, scales_ptr, offsets_ptr, output_ptr, n);
      }
      if (m > 32) {
        if (offsets.empty())
          return blkq4_gemm_sm80<ElementT, 32, false, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<ElementT, 32, false, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<ElementT, 32, false, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<ElementT, 32, false, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    }
    break;

  case 64:
    if (column_wise_blocking) {
      if (m <= 64 && n < 16384) {
        if (offsets.empty())
          return blkq4_small_gemm_sm80<ElementT, 64, true, false>(m, n, k, stream, a_ptr, k, weights_ptr, scales_ptr, offsets_ptr, output_ptr, n);
        else
          return blkq4_small_gemm_sm80<ElementT, 64, true, true>(m, n, k, stream, a_ptr, k, weights_ptr, scales_ptr, offsets_ptr, output_ptr, n);
      }
      if (m > 32) {
        if (offsets.empty())
          return blkq4_gemm_sm80<ElementT, 64, true, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<ElementT, 64, true, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<ElementT, 64, true, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<ElementT, 64, true, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    } else {
      if (m <= 64 && n < 16384) {
        if (offsets.empty())
          return blkq4_small_gemm_sm80<ElementT, 64, false, false>(m, n, k, stream, a_ptr, k, weights_ptr, scales_ptr, offsets_ptr, output_ptr, n);
        else
          return blkq4_small_gemm_sm80<ElementT, 64, false, true>(m, n, k, stream, a_ptr, k, weights_ptr, scales_ptr, offsets_ptr, output_ptr, n);
      }
      if (m > 32) {
        if (offsets.empty())
          return blkq4_gemm_sm80<ElementT, 64, false, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<ElementT, 64, false, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<ElementT, 64, false, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<ElementT, 64, false, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    }
    break;
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported block size: ", block_size);
}

template
Status blkq4_fp16_gemm_sm80_dispatch<half>(
  int block_size,
  bool column_wise_blocking,
  int m, int n, int k, cudaStream_t stream,
  half const* a_ptr, size_t a_size,
  uint8_t const* weights_ptr, size_t weights_size,
  half const* scales_ptr, size_t scales_size,
  uint8_t const* offsets_ptr, size_t offsets_size,
  half* output_ptr, size_t output_size);

template
Status blkq4_fp16_gemm_sm80_dispatch<cutlass::half_t>(
  int block_size,
  bool column_wise_blocking,
  int m, int n, int k, cudaStream_t stream,
  cutlass::half_t const* a_ptr, size_t a_size,
  uint8_t const* weights_ptr, size_t weights_size,
  cutlass::half_t const* scales_ptr, size_t scales_size,
  uint8_t const* offsets_ptr, size_t offsets_size,
  cutlass::half_t* output_ptr, size_t output_size);

template
Status blkq4_fp16_gemm_sm80_dispatch<onnxruntime::MLFloat16>(
  int block_size, bool column_wise_blocking, int m, int n, int k, cudaStream_t stream,
  onnxruntime::MLFloat16 const* a_ptr, size_t a_size,
  uint8_t const* weights_ptr, size_t weights_size,
  onnxruntime::MLFloat16 const* scales_ptr, size_t scales_size,
  uint8_t const* offsets_ptr, size_t offsets_size,
  onnxruntime::MLFloat16* output_ptr, size_t output_size);

#endif  // !defined(USE_MIGRAPHX) && !defined(USE_ROCM)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
