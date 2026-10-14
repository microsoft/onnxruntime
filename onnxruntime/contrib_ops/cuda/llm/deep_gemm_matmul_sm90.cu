// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8.h"

#include <algorithm>
#include <cuda.h>
#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {
namespace {
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kBlockK = 128;
constexpr int kStages = 8;

int Blocks(size_t count) {
  return static_cast<int>(std::min<size_t>((count + 255) / 256, 65535));
}

template <typename T>
__global__ void QuantizeActivation(const T* input, __nv_fp8_e4m3* output,
                                   const float* scale, float* scales, size_t count, size_t scale_count) {
  const float value = *scale;
  const float inv_scale = value != 0.0f ? 1.0f / value : 0.0f;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    output[i] = __nv_fp8_e4m3(static_cast<float>(input[i]) * inv_scale);
    if (i < scale_count) {
      scales[i] = value;
    }
  }
}

// SFB is logically [N, K/128], physically [K/128, N] for TMA.
__global__ void PackWeightScales(const float* input, float* output, int n, int k_blocks) {
  const size_t count = static_cast<size_t>(n) * k_blocks;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t row = i % n;
    output[i] = input[row * k_blocks + i / n];
  }
}

template <typename T>
__global__ void ConvertOutput(const float* input, T* output, const T* bias,
                              int m, int n, int output_stride) {
  const size_t count = static_cast<size_t>(m) * n;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    // The existing cuBLAS path rounds to T before adding bias.
    T value = static_cast<T>(input[i]);
    if (bias != nullptr) {
      value = static_cast<T>(static_cast<float>(value) + static_cast<float>(bias[i % n]));
    }
    output[(i / n) * output_stride + i % n] = value;
  }
}

Status TensorMap(CUtensorMap& map, const void* data, CUtensorMapDataType dtype,
                 uint32_t element_size, uint32_t inner, uint32_t outer,
                 uint32_t box_inner, uint32_t box_outer, bool swizzle) {
  const cuuint64_t dims[] = {inner, outer};
  const cuuint64_t strides[] = {static_cast<cuuint64_t>(inner) * element_size};
  const cuuint32_t box[] = {box_inner, box_outer};
  const cuuint32_t element_strides[] = {1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, dtype, 2, const_cast<void*>(data), dims, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  ORT_RETURN_IF_NOT(result == CUDA_SUCCESS, "DeepGEMM MatMul tensor map failed: ", static_cast<int>(result));
  return Status::OK();
}

template <int NumSms>
Status Launch(const CUtensorMap& a, const CUtensorMap& b, const CUtensorMap& sfa,
              const CUtensorMap& sfb, const CUtensorMap& d, int m, int n, int k, cudaStream_t stream) {
  auto kernel = &deep_gemm::sm90_fp8_gemm_1d1d_impl<
      0, 0, 0, 1, kBlockM, kBlockN, kBlockK, 128, 128, kStages,
      128, 128, 1, false, NumSms, deep_gemm::GemmType::Normal, float>;
  // D, pipelined A/B/SFA/SFB, and two 8-byte barriers per stage.
  constexpr int aligned_sfb_bytes = (kBlockN * sizeof(float) + 127) / 128 * 128;
  constexpr int kSmemBytes = kBlockM * kBlockN * sizeof(float) +
                             kStages * ((kBlockM + kBlockN) * kBlockK +
                                        kBlockM * sizeof(float) + aligned_sfb_bytes + 16);
  static_assert(kSmemBytes <= 232448);
  CUDA_RETURN_IF_ERROR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim = {1, 1, 1};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(NumSms);
  config.blockDim = dim3(256);
  config.dynamicSmemBytes = kSmemBytes;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  CUDA_RETURN_IF_ERROR(cudaLaunchKernelEx(
      &config, kernel, static_cast<__nv_fp8_e4m3*>(nullptr), static_cast<__nv_fp8_e4m3*>(nullptr),
      static_cast<int*>(nullptr), static_cast<CUtensorMap*>(nullptr),
      static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k), a, b, sfa, sfb, d));
  return Status::OK();
}

Status Dispatch(const CUtensorMap& a, const CUtensorMap& b, const CUtensorMap& sfa,
                const CUtensorMap& sfb, const CUtensorMap& d, int m, int n, int k,
                int sm_count, cudaStream_t stream) {
  // The scheduler's compile-time stride must equal the launched grid size.
  if (sm_count >= 132) {
    return Launch<132>(a, b, sfa, sfb, d, m, n, k, stream);
  }
  if (sm_count >= 114) {
    return Launch<114>(a, b, sfa, sfb, d, m, n, k, stream);
  }
  if (sm_count >= 64) {
    return Launch<64>(a, b, sfa, sfb, d, m, n, k, stream);
  }
  if (sm_count >= 32) {
    return Launch<32>(a, b, sfa, sfb, d, m, n, k, stream);
  }
  return Launch<16>(a, b, sfa, sfb, d, m, n, k, stream);
}
}  // namespace

Status LaunchPrepareMatMulFp8DeepGemm(void* a_quant, float* a_scales, const void* a,
                                      const float* a_scale, int m, int k, int aligned_m,
                                      bool is_bf16, cudaStream_t stream) {
  const size_t count = SafeInt<size_t>(m) * k;
  const size_t scale_count = SafeInt<size_t>(aligned_m) * (k / kBlockK);
  if (is_bf16) {
    QuantizeActivation<<<Blocks(count), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(a), static_cast<__nv_fp8_e4m3*>(a_quant), a_scale, a_scales, count, scale_count);
  } else {
    QuantizeActivation<<<Blocks(count), 256, 0, stream>>>(
        static_cast<const half*>(a), static_cast<__nv_fp8_e4m3*>(a_quant), a_scale, a_scales, count, scale_count);
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

Status LaunchMatMulFp8DeepGemm(void* y, const void* a_quant, const float* a_scales,
                               const void* b, const float* b_scales, const void* bias,
                               float* packed_b_scales, float* accum, int m, int n, int k,
                               int aligned_m, int output_stride, bool is_bf16,
                               int sm_count, cudaStream_t stream) {
  const size_t count = SafeInt<size_t>(m) * n;
  const size_t scale_count = SafeInt<size_t>(n) * (k / kBlockK);
  PackWeightScales<<<Blocks(scale_count), 256, 0, stream>>>(b_scales, packed_b_scales, n, k / kBlockK);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  // The upstream kernel uses TMA reduce-add, including for an ordinary dense GEMM.
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(accum, 0, SafeInt<size_t>(count) * sizeof(float), stream));
  CUtensorMap map_a{}, map_b{}, map_sfa{}, map_sfb{}, map_d{};
  ORT_RETURN_IF_ERROR(TensorMap(map_a, a_quant, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, k, m, 128, kBlockM, true));
  ORT_RETURN_IF_ERROR(TensorMap(map_b, b, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, k, n, 128, kBlockN, true));
  ORT_RETURN_IF_ERROR(TensorMap(map_sfa, a_scales, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
                                aligned_m, k / kBlockK, kBlockM, 1, false));
  ORT_RETURN_IF_ERROR(TensorMap(map_sfb, packed_b_scales, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
                                n, k / kBlockK, kBlockN, 1, false));
  ORT_RETURN_IF_ERROR(TensorMap(map_d, accum, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, n, m, kBlockN, kBlockM, false));
  ORT_RETURN_IF_ERROR(Dispatch(map_a, map_b, map_sfa, map_sfb, map_d, m, n, k, sm_count, stream));
  if (is_bf16) {
    ConvertOutput<<<Blocks(count), 256, 0, stream>>>(
        accum, static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(bias), m, n, output_stride);
  } else {
    ConvertOutput<<<Blocks(count), 256, 0, stream>>>(
        accum, static_cast<half*>(y), static_cast<const half*>(bias), m, n, output_stride);
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}
}  // namespace onnxruntime::contrib::cuda
