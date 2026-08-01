// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp4.h"

#include "core/providers/cuda/cuda_common.h"

#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/half.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace onnxruntime::contrib::cuda {
namespace {

using namespace cute;

// e4m3_to_float / float_to_e4m3 / Vec2Traits come from cuda_type_helper.cuh.

constexpr int kScaleVectorSize = 16;
constexpr int kAlignment = 32;
constexpr int kQuantizeThreadsPerBlock = 128;

__device__ __forceinline__ int SwizzledScaleOffset(int row, int k_block, int num_k_tiles) {
  const int row_tile = row >> 7;
  const int outer_row = row & 31;
  const int inner_row = (row >> 5) & 3;
  const int k_tile = k_block >> 2;
  const int inner_k = k_block & 3;
  return ((row_tile * num_k_tiles + k_tile) << 9) | (outer_row << 4) | (inner_row << 2) | inner_k;
}

// The activation global scale divides both `alpha` and every per-block scale, so a malformed value
// would poison the whole GEMM. Fall back to 1 (i.e. "no global scale") when the caller did not
// supply one or supplied zero, a denormal, a negative number, or a NaN/Inf.
__device__ __forceinline__ float SafeActivationGlobalScale(const float* input_scale) {
  if (input_scale == nullptr) {
    return 1.0f;
  }
  const float scale = input_scale[0];
  // NaN fails both comparisons, +/-Inf fails the upper bound, and zero/denormals/negatives fail
  // the lower bound, so every malformed value falls back to 1.
  return (scale >= FLT_MIN && scale <= FLT_MAX) ? scale : 1.0f;
}

// Quantizes the FP16/BF16 activation [m, k] to NVFP4 with one E4M3 scale per 16-element block.
//
// One thread owns one 16-element block: it issues two 16-byte loads for its slice, reduces the
// block maximum in registers, and writes the 8 packed FP4 bytes back with a single 8-byte store.
// grid = (ceil(k_blocks / kQuantizeThreadsPerBlock), m), so consecutive threads cover consecutive
// K blocks of the same row and the loads/stores coalesce across the warp.
template <typename T>
__global__ void QuantizeActivationNvFp4Kernel(const T* __restrict__ a,
                                              const float* __restrict__ input_scale,
                                              uint8_t* __restrict__ a_packed,
                                              uint8_t* __restrict__ a_scale,
                                              float* __restrict__ alpha,
                                              const float* __restrict__ weight_scale_2,
                                              int m,
                                              int k,
                                              int k_blocks,
                                              int rounded_k_blocks) {
  const float activation_global_scale = SafeActivationGlobalScale(input_scale);
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    alpha[0] = weight_scale_2[0] / activation_global_scale;
  }

  const int row = static_cast<int>(blockIdx.y);
  const int k_block = static_cast<int>(blockIdx.x) * kQuantizeThreadsPerBlock + static_cast<int>(threadIdx.x);
  if (row >= m || k_block >= k_blocks) {
    return;
  }

  using Traits = Vec2Traits<T>;
  using T2 = typename Traits::Type2;
  constexpr int kPairs = kScaleVectorSize / 2;

  // 16 elements of a 2-byte type = 32 bytes = two uint4 loads. The launcher enforces
  // k % kAlignment == 0 (and kAlignment is a multiple of kScaleVectorSize), so
  // (row * k + k_block * 16) * sizeof(T) is always a multiple of 16 bytes.
  const uint4* a_vec = reinterpret_cast<const uint4*>(a + static_cast<size_t>(row) * k + k_block * kScaleVectorSize);
  uint4 raw_lo = a_vec[0];
  uint4 raw_hi = a_vec[1];
  const T2* raw_pairs[2] = {reinterpret_cast<const T2*>(&raw_lo), reinterpret_cast<const T2*>(&raw_hi)};

  float2 values[kPairs];
  float max_abs = 0.0f;
#pragma unroll
  for (int pair = 0; pair < kPairs; ++pair) {
    const float2 value = Traits::to_float2(raw_pairs[pair >> 2][pair & 3]);
    values[pair] = value;
    max_abs = fmaxf(max_abs, fmaxf(fabsf(value.x), fabsf(value.y)));
  }

  const uint8_t raw_scale = float_to_e4m3(fmaxf(max_abs / 6.0f, 1.0f / 1024.0f) * activation_global_scale);
  a_scale[SwizzledScaleOffset(row, k_block, rounded_k_blocks / 4)] = raw_scale;
  const float local_scale = e4m3_to_float(raw_scale) / activation_global_scale;

  uint2 packed;
  packed.x = 0;
  packed.y = 0;
#pragma unroll
  for (int pair = 0; pair < kPairs; ++pair) {
    const float2 scaled = make_float2(values[pair].x / local_scale, values[pair].y / local_scale);
    const uint32_t byte = static_cast<uint32_t>(
        static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest)));
    if (pair < 4) {
      packed.x |= byte << (8 * pair);
    } else {
      packed.y |= byte << (8 * (pair - 4));
    }
  }
  // 8 packed bytes per 16-element block; k % kAlignment == 0 keeps this 8-byte aligned.
  *reinterpret_cast<uint2*>(a_packed + static_cast<size_t>(row) * (k / 2) + k_block * kPairs) = packed;
}

__global__ void RepackWeightScaleNvFp4Kernel(const uint8_t* __restrict__ weight_scale,
                                             uint8_t* __restrict__ b_scale,
                                             int n,
                                             int k_blocks,
                                             int rounded_k_blocks) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = n * rounded_k_blocks;
  if (index >= total) {
    return;
  }

  const int row = index / rounded_k_blocks;
  const int k_block = index - row * rounded_k_blocks;
  const uint8_t scale = k_block < k_blocks ? weight_scale[row * k_blocks + k_block] : 0;
  b_scale[SwizzledScaleOffset(row, k_block, rounded_k_blocks / 4)] = scale;
}

struct Fp4GemmSm120M256Config {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = void;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape = Shape<_128, _128, _128>;
};

struct Fp4GemmSm120DefaultConfig {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = cutlass::gemm::PersistentScheduler;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape = Shape<_256, _128, _128>;
};

template <typename Config, typename OutType>
struct Fp4GemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = OutType;
  using ElementD = OutType;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, typename Config::PerSmTileShape, typename Config::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 128 / cutlass::sizeof_bits<ElementC>::value,
      ElementD, LayoutD, 128 / cutlass::sizeof_bits<ElementD>::value,
      typename Config::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, kAlignment,
      ElementB, LayoutB, kAlignment,
      ElementAccumulator, typename Config::MmaTileShape, typename Config::ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename Config::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, typename Config::TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
typename Gemm::Arguments MakeArguments(void* y,
                                       const void* a_packed,
                                       const void* b_packed,
                                       const void* a_scale,
                                       const void* b_scale,
                                       const float* alpha,
                                       int m,
                                       int n,
                                       int k) {
  using ElementA = typename Gemm::GemmKernel::ElementA;
  using ElementB = typename Gemm::GemmKernel::ElementB;
  using ElementD = typename Gemm::GemmKernel::ElementD;
  using ElementSF = cutlass::float_ue4m3_t;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  constexpr int l = 1;
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));
  auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, l));
  auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, l));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {reinterpret_cast<const ElementA*>(a_packed), stride_a,
       reinterpret_cast<const ElementB*>(b_packed), stride_b,
       reinterpret_cast<const ElementSF*>(a_scale), layout_sfa,
       reinterpret_cast<const ElementSF*>(b_scale), layout_sfb},
      {{}, reinterpret_cast<ElementD*>(y), stride_d, reinterpret_cast<ElementD*>(y), stride_d}};
  arguments.epilogue.thread.alpha_ptr = alpha;
  return arguments;
}

template <typename Gemm>
size_t WorkspaceSize(int m, int n, int k) {
  auto arguments = MakeArguments<Gemm>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k);
  return Gemm::get_workspace_size(arguments);
}

template <typename Gemm>
Status RunGemm(void* y,
               const void* a_packed,
               const void* b_packed,
               const void* a_scale,
               const void* b_scale,
               const float* alpha,
               int m,
               int n,
               int k,
               void* workspace,
               cudaStream_t stream) {
  auto arguments = MakeArguments<Gemm>(y, a_packed, b_packed, a_scale, b_scale, alpha, m, n, k);
  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess,
                    "SM120 native FP4 GEMM cannot implement the given problem: ",
                    cutlassGetStatusString(status));
  status = gemm.initialize(arguments, workspace, stream);
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess,
                    "SM120 native FP4 GEMM initialize failed: ", cutlassGetStatusString(status));
  status = gemm.run(arguments, workspace, stream);
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess,
                    "SM120 native FP4 GEMM run failed: ", cutlassGetStatusString(status));
  return CUDA_CALL(cudaGetLastError());
}

bool UseM256Config(int m) {
  // Smallest power of two >= m, computed portably (no compiler builtins so this
  // also compiles under MSVC host compilation for CUDA builds on Windows).
  int next_power_of_two_m = 1;
  while (next_power_of_two_m < m) {
    next_power_of_two_m <<= 1;
  }
  return std::max(16, next_power_of_two_m) <= 256;
}

template <typename OutType>
size_t DispatchWorkspaceSize(int m, int n, int k) {
  if (UseM256Config(m)) {
    return WorkspaceSize<typename Fp4GemmSm120<Fp4GemmSm120M256Config, OutType>::Gemm>(m, n, k);
  }
  return WorkspaceSize<typename Fp4GemmSm120<Fp4GemmSm120DefaultConfig, OutType>::Gemm>(m, n, k);
}

template <typename OutType>
Status DispatchRunGemm(void* y,
                       const void* a_packed,
                       const void* b_packed,
                       const void* a_scale,
                       const void* b_scale,
                       const float* alpha,
                       int m,
                       int n,
                       int k,
                       void* workspace,
                       cudaStream_t stream) {
  if (UseM256Config(m)) {
    return RunGemm<typename Fp4GemmSm120<Fp4GemmSm120M256Config, OutType>::Gemm>(
        y, a_packed, b_packed, a_scale, b_scale, alpha, m, n, k, workspace, stream);
  }
  return RunGemm<typename Fp4GemmSm120<Fp4GemmSm120DefaultConfig, OutType>::Gemm>(
      y, a_packed, b_packed, a_scale, b_scale, alpha, m, n, k, workspace, stream);
}

}  // namespace

size_t GetMatMulBlockQuantizedFp4WeightNativeSm120WorkspaceSize(int m, int n, int k, bool is_bf16) {
  return is_bf16 ? DispatchWorkspaceSize<cutlass::bfloat16_t>(m, n, k)
                 : DispatchWorkspaceSize<cutlass::half_t>(m, n, k);
}

Status LaunchRepackWeightScaleNvFp4ForNativeSm120(void* b_scale,
                                                  const void* weight_scale,
                                                  int n,
                                                  int k,
                                                  int block_size,
                                                  cudaStream_t stream) {
  ORT_RETURN_IF_NOT(block_size == kScaleVectorSize,
                    "SM120 native FP4 GEMM only supports block_size == ", kScaleVectorSize);
  ORT_RETURN_IF_NOT(k % kAlignment == 0, "SM120 native FP4 GEMM requires K divisible by ", kAlignment);
  ORT_RETURN_IF_NOT(n % kAlignment == 0, "SM120 native FP4 GEMM requires N divisible by ", kAlignment);

  const int k_blocks = k / kScaleVectorSize;
  const int rounded_k_blocks = ((k_blocks + 3) / 4) * 4;
  constexpr int kThreads = 256;
  const int repack_total = n * rounded_k_blocks;
  const int repack_blocks = (repack_total + kThreads - 1) / kThreads;
  RepackWeightScaleNvFp4Kernel<<<repack_blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(weight_scale), reinterpret_cast<uint8_t*>(b_scale), n, k_blocks,
      rounded_k_blocks);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchMatMulBlockQuantizedFp4WeightNativeSm120(void* y,
                                                      const void* a,
                                                      const void* b_packed,
                                                      const float* weight_scale_2,
                                                      const float* input_scale,
                                                      void* a_packed,
                                                      void* a_scale,
                                                      const void* b_scale,
                                                      float* alpha,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      int block_size,
                                                      bool is_bf16,
                                                      void* workspace,
                                                      size_t workspace_size,
                                                      cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(workspace_size);
  ORT_RETURN_IF_NOT(block_size == kScaleVectorSize,
                    "SM120 native FP4 GEMM only supports block_size == ", kScaleVectorSize);
  ORT_RETURN_IF_NOT(k % kAlignment == 0, "SM120 native FP4 GEMM requires K divisible by ", kAlignment);
  ORT_RETURN_IF_NOT(n % kAlignment == 0, "SM120 native FP4 GEMM requires N divisible by ", kAlignment);

  const int k_blocks = k / kScaleVectorSize;
  const int rounded_k_blocks = ((k_blocks + 3) / 4) * 4;
  const dim3 quant_grid{static_cast<unsigned int>((k_blocks + kQuantizeThreadsPerBlock - 1) / kQuantizeThreadsPerBlock),
                        static_cast<unsigned int>(m)};
  if (is_bf16) {
    QuantizeActivationNvFp4Kernel<nv_bfloat16><<<quant_grid, kQuantizeThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(a), input_scale, reinterpret_cast<uint8_t*>(a_packed),
        reinterpret_cast<uint8_t*>(a_scale), alpha, weight_scale_2, m, k, k_blocks, rounded_k_blocks);
  } else {
    QuantizeActivationNvFp4Kernel<half><<<quant_grid, kQuantizeThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const half*>(a), input_scale, reinterpret_cast<uint8_t*>(a_packed),
        reinterpret_cast<uint8_t*>(a_scale), alpha, weight_scale_2, m, k, k_blocks, rounded_k_blocks);
  }
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

  if (is_bf16) {
    return DispatchRunGemm<cutlass::bfloat16_t>(
        y, a_packed, b_packed, a_scale, b_scale, alpha, m, n, k, workspace, stream);
  }
  return DispatchRunGemm<cutlass::half_t>(
      y, a_packed, b_packed, a_scale, b_scale, alpha, m, n, k, workspace, stream);
}

}  // namespace onnxruntime::contrib::cuda
