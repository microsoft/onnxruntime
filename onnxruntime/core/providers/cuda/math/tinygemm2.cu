// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/tinygemm2.h"

#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "core/providers/cuda/math/tinygemm2_kernel.cuh"

namespace onnxruntime {
namespace cuda {
namespace {

constexpr int64_t kMaxM = 64;
constexpr int64_t kMaxWeightElements = int64_t{32} << 20;

template <typename T>
constexpr CUtensorMapDataType TensorMapDataType() {
  return std::is_same_v<T, half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
}

// Rank-2 row-major tensor map: `inner` contiguous elements per row, `outer` rows.
template <typename T>
Status EncodeTensorMap(CUtensorMap& map, const T* data, uint64_t inner, uint64_t outer, uint32_t box_inner,
                       uint32_t box_outer, CUtensorMapSwizzle swizzle) {
  const cuuint64_t dims[] = {inner, outer};
  const cuuint64_t strides[] = {inner * sizeof(T)};
  const cuuint32_t box[] = {box_inner, box_outer};
  const cuuint32_t element_strides[] = {1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, TensorMapDataType<T>(), 2, const_cast<T*>(data), dims, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  ORT_RETURN_IF_NOT(result == CUDA_SUCCESS, "tinygemm2 cuTensorMapEncodeTiled failed: ", static_cast<int>(result));
  return Status::OK();
}

template <typename T>
bool PrepareKernel(const cudaDeviceProp& device_prop) {
  cudaFuncAttributes attributes{};
  if (cudaFuncGetAttributes(&attributes, tinygemm2::TinyGemm2Kernel<T>) != cudaSuccess) {
    return false;
  }
  // Code generated from pre-SM 9.0 PTX (including a JIT for this device) has an empty kernel body.
  if (attributes.ptxVersion < 90 ||
      attributes.sharedSizeBytes + tinygemm2::kDynamicSmemBytes > device_prop.sharedMemPerBlockOptin) {
    return false;
  }
  return cudaFuncSetAttribute(tinygemm2::TinyGemm2Kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              tinygemm2::kDynamicSmemBytes) == cudaSuccess;
}

}  // namespace

bool CanUseTinyGemm2(int64_t m, int64_t n, int64_t k, const void* a, const void* b) {
  if (m < 1 || m > kMaxM || n < 1 || k < 1 || n > kMaxWeightElements / k) return false;
  if (n % 8 != 0 || k % 8 != 0) return false;
  return ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) % 16) == 0;
}

bool IsTinyGemm2Supported(const cudaDeviceProp& device_prop) {
  if (device_prop.major < 9) {
    return false;
  }
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  static std::mutex mutex;
  static std::unordered_map<int, bool> supported;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = supported.find(device);
  if (it == supported.end()) {
    const bool ok = PrepareKernel<half>(device_prop) && PrepareKernel<nv_bfloat16>(device_prop);
    // Clear any error from probing so it is not reported by an unrelated later call.
    cudaGetLastError();
    it = supported.emplace(device, ok).first;
  }
  return it->second;
}

template <typename T>
Status LaunchTinyGemm2(cudaStream_t stream, const T* a, const T* b, T* c, int m, int n, int k, bool b_is_constant) {
  using namespace tinygemm2;
  ORT_RETURN_IF_NOT(CanUseTinyGemm2(m, n, k, a, b), "tinygemm2 does not support M=", m, " N=", n, " K=", k, ".");
  alignas(128) TensorMaps maps{};
  auto& weight_map = *reinterpret_cast<CUtensorMap*>(maps.weight);
  auto& activation_map = *reinterpret_cast<CUtensorMap*>(maps.activation);
  // B is [K, N]: kTileN columns (32 bytes) by kTileK rows per box.
  ORT_RETURN_IF_ERROR(EncodeTensorMap(weight_map, b, static_cast<uint64_t>(n), static_cast<uint64_t>(k), kTileN,
                                      kTileK, CU_TENSOR_MAP_SWIZZLE_32B));
  // A is [M, K]: kTileK columns (128 bytes) by kTileM rows per box.
  ORT_RETURN_IF_ERROR(EncodeTensorMap(activation_map, a, static_cast<uint64_t>(k), static_cast<uint64_t>(m), kTileK,
                                      kTileM, CU_TENSOR_MAP_SWIZZLE_128B));

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>((n + kTileN - 1) / kTileN), static_cast<unsigned>((m + kTileM - 1) / kTileM));
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = stream;
#if CUDART_VERSION >= 12030
  // CUDA graphs capture programmatic dependent launch from CUDA 12.3.
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  if (b_is_constant) {
    config.attrs = &attribute;
    config.numAttrs = 1;
  }
#else
  ORT_UNUSED_PARAMETER(b_is_constant);
#endif
  CUDA_RETURN_IF_ERROR(cudaLaunchKernelEx(&config, TinyGemm2Kernel<T>, maps, c, m, n, k));
  return Status::OK();
}

template Status LaunchTinyGemm2<half>(cudaStream_t, const half*, const half*, half*, int, int, int, bool);
template Status LaunchTinyGemm2<nv_bfloat16>(cudaStream_t, const nv_bfloat16*, const nv_bfloat16*, nv_bfloat16*,
                                             int, int, int, bool);

}  // namespace cuda
}  // namespace onnxruntime
