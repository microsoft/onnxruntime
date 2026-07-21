// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp4.h"

#include "core/providers/cuda/cuda_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

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

constexpr int kScaleVectorSize = 16;
constexpr int kAlignment = 32;

template <typename T>
__device__ __forceinline__ float LoadAsFloat(const T* data, int index);

template <>
__device__ __forceinline__ float LoadAsFloat<half>(const half* data, int index) {
  return __half2float(data[index]);
}

template <>
__device__ __forceinline__ float LoadAsFloat<nv_bfloat16>(const nv_bfloat16* data, int index) {
  return __bfloat162float(data[index]);
}

__device__ __forceinline__ int SwizzledScaleOffset(int row, int k_block, int num_k_tiles) {
  const int row_tile = row >> 7;
  const int outer_row = row & 31;
  const int inner_row = (row >> 5) & 3;
  const int k_tile = k_block >> 2;
  const int inner_k = k_block & 3;
  return ((row_tile * num_k_tiles + k_tile) << 9) | (outer_row << 4) | (inner_row << 2) | inner_k;
}

__device__ __forceinline__ uint8_t FloatToE4m3(float value) {
  __nv_fp8_e4m3 converted(value);
  uint8_t raw;
  reinterpret_cast<__nv_fp8_e4m3&>(raw) = converted;
  return raw;
}

__device__ __forceinline__ float E4m3ToFloat(uint8_t raw) {
  return __half2float(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(raw), __NV_E4M3));
}

template <typename T>
__global__ void QuantizeActivationNvFp4Kernel(const T* __restrict__ a,
                                              const float* __restrict__ input_scale,
                                              uint8_t* __restrict__ a_packed,
                                              uint8_t* __restrict__ a_scale,
                                              float* __restrict__ alpha,
                                              const float* __restrict__ weight_scale_2,
                                              int m,
                                              int k,
                                              int rounded_k_blocks) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const float activation_global_scale = input_scale != nullptr ? input_scale[0] : 1.0f;
    alpha[0] = weight_scale_2[0] / activation_global_scale;
  }

  const int row = static_cast<int>(blockIdx.y);
  const int k_block = static_cast<int>(blockIdx.x);
  if (row >= m || k_block >= k / kScaleVectorSize) {
    return;
  }

  const int k_base = k_block * kScaleVectorSize;
  float values[kScaleVectorSize];
  float max_abs = 0.0f;
#pragma unroll
  for (int offset = 0; offset < kScaleVectorSize; ++offset) {
    const float value = LoadAsFloat<T>(a, row * k + k_base + offset);
    values[offset] = value;
    max_abs = fmaxf(max_abs, fabsf(value));
  }

  const float activation_global_scale = input_scale != nullptr ? input_scale[0] : 1.0f;
  const uint8_t raw_scale = FloatToE4m3(fmaxf(max_abs / 6.0f, 1.0f / 1024.0f) * activation_global_scale);
  a_scale[SwizzledScaleOffset(row, k_block, rounded_k_blocks / 4)] = raw_scale;
  const float local_scale = E4m3ToFloat(raw_scale) / activation_global_scale;

#pragma unroll
  for (int pair = 0; pair < kScaleVectorSize / 2; ++pair) {
    const float2 scaled = make_float2(values[pair * 2] / local_scale, values[pair * 2 + 1] / local_scale);
    a_packed[row * (k / 2) + k_base / 2 + pair] =
        static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
  }
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
  const auto m_unsigned = static_cast<unsigned int>(std::max(m - 1, 1));
  const int next_power_of_two_m = static_cast<int>(1u << (32 - __builtin_clz(m_unsigned)));
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

size_t GetMatMulBlockScaledFp4NativeSm120WorkspaceSize(int m, int n, int k, bool is_bf16) {
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

Status LaunchMatMulBlockScaledFp4NativeSm120(void* y,
                                             const void* a,
                                             const void* b_packed,
                                             const void* weight_scale,
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
  ORT_UNUSED_PARAMETER(weight_scale);
  ORT_RETURN_IF_NOT(block_size == kScaleVectorSize,
                    "SM120 native FP4 GEMM only supports block_size == ", kScaleVectorSize);
  ORT_RETURN_IF_NOT(k % kAlignment == 0, "SM120 native FP4 GEMM requires K divisible by ", kAlignment);
  ORT_RETURN_IF_NOT(n % kAlignment == 0, "SM120 native FP4 GEMM requires N divisible by ", kAlignment);

  const int k_blocks = k / kScaleVectorSize;
  const int rounded_k_blocks = ((k_blocks + 3) / 4) * 4;
  const dim3 quant_grid{static_cast<unsigned int>(k_blocks), static_cast<unsigned int>(m)};
  if (is_bf16) {
    QuantizeActivationNvFp4Kernel<nv_bfloat16><<<quant_grid, 1, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(a), input_scale, reinterpret_cast<uint8_t*>(a_packed),
        reinterpret_cast<uint8_t*>(a_scale), alpha, weight_scale_2, m, k, rounded_k_blocks);
  } else {
    QuantizeActivationNvFp4Kernel<half><<<quant_grid, 1, 0, stream>>>(
        reinterpret_cast<const half*>(a), input_scale, reinterpret_cast<uint8_t*>(a_packed),
        reinterpret_cast<uint8_t*>(a_scale), alpha, weight_scale_2, m, k, rounded_k_blocks);
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
