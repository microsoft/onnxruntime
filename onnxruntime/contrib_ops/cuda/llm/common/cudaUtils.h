/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "contrib_ops/cuda/llm/common/cudaBf16Wrapper.h"
#include "contrib_ops/cuda/llm/common/cudaDriverWrapper.h"
#include "contrib_ops/cuda/llm/common/cudaFp8Utils.h"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/common/tllmException.h"
#include <algorithm>
#include <cinttypes>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#ifndef _WIN32  // Linux
#include <sys/sysinfo.h>
#endif  // not WIN32
#include <vector>
#ifdef _WIN32  // Windows
#include <windows.h>
#undef ERROR  // A Windows header file defines ERROR as 0, but it's used in our logger.h enum. Logging breaks without
              // this undef.
#endif        // WIN32

namespace onnxruntime::llm::common {

// workspace for cublas gemm : 32MB
#define CUBLAS_WORKSPACE_SIZE 33554432

typedef struct __align__(4) {
  half x, y, z, w;
}

half4;

/* **************************** type definition ***************************** */

enum CublasDataType {
  FLOAT_DATATYPE = 0,
  HALF_DATATYPE = 1,
  BFLOAT16_DATATYPE = 2,
  INT8_DATATYPE = 3,
  FP8_DATATYPE = 4
};

enum TRTLLMCudaDataType {
  FP32 = 0,
  FP16 = 1,
  BF16 = 2,
  INT8 = 3,
  FP8 = 4
};

enum class OperationType {
  FP32,
  FP16,
  BF16,
  INT8,
  FP8
};

/* **************************** debug tools ********************************* */
static char const* _cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorString(error);
}

static char const* _cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void check(T ptr, char const* const func, char const* const file, int const line) {
  if (ptr) {
    throw TllmException(
        file, line, fmtstr("[TensorRT-LLM][ERROR] CUDA runtime error in %s: %s", func, _cudaGetErrorEnum(ptr)));
  }
}

template <typename T>
void checkEx(
    T ptr, std::initializer_list<T> const& validReturns, char const* const func, char const* const file, int const line) {
  if (std::all_of(std::begin(validReturns), std::end(validReturns), [&ptr](T const& t) { return t != ptr; })) {
    throw TllmException(
        file, line, fmtstr("[TensorRT-LLM][ERROR] CUDA runtime error in %s: %s", func, _cudaGetErrorEnum(ptr)));
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) check((val), #val, file, line)

inline std::optional<bool> isCudaLaunchBlocking() {
  static bool firstCall = true;
  static std::optional<bool> ptr = std::nullopt;

  if (firstCall) {
    char const* env = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (env != nullptr && std::string(env) == "1") {
      ptr = true;
    } else if (env != nullptr && std::string(env) == "0") {
      ptr = false;
    }
    firstCall = false;
  }

  return ptr;
}

inline bool doCheckError() {
  auto const cudaLaunchBlocking = isCudaLaunchBlocking();
#ifndef NDEBUG
  bool const checkError = cudaLaunchBlocking.value_or(true);
#else
  bool const checkError = cudaLaunchBlocking.value_or(false);
#endif

  return checkError;
}

inline void syncAndCheck(char const* const file, int const line) {
  if (doCheckError()) {
    cudaDeviceSynchronize();
    check(cudaGetLastError(), "cudaGetLastError", file, line);
  }
}

#define sync_check_cuda_error() onnxruntime::llm::common::syncAndCheck(__FILE__, __LINE__)

#define PRINT_FUNC_NAME_()                                                    \
  do {                                                                        \
    std::cout << "[TensorRT-LLM][CALL] " << __FUNCTION__ << " " << std::endl; \
  } while (0)

// clang-format off
template<typename T> struct packed_type;
template <>          struct packed_type<float>         { using type = float; }; // we don't need to pack float by default
template <>          struct packed_type<half>          { using type = half2; };

#ifdef ENABLE_BF16
template<>
struct packed_type<__nv_bfloat16> {
    using type = __nv_bfloat162;
};
#endif

#ifdef ENABLE_FP8
template<>
struct packed_type<__nv_fp8_e4m3> {
    using type = __nv_fp8x2_e4m3;
};
#endif

template<typename T> struct num_elems;
template <>          struct num_elems<float>           { static constexpr int value = 1; };
template <>          struct num_elems<float2>          { static constexpr int value = 2; };
template <>          struct num_elems<float4>          { static constexpr int value = 4; };
template <>          struct num_elems<half>            { static constexpr int value = 1; };
template <>          struct num_elems<half2>           { static constexpr int value = 2; };
#ifdef ENABLE_BF16
template <>          struct num_elems<__nv_bfloat16>   { static constexpr int value = 1; };
template <>          struct num_elems<__nv_bfloat162>  { static constexpr int value = 2; };
#endif
#ifdef ENABLE_FP8
template <>          struct num_elems<__nv_fp8_e4m3>   { static constexpr int value = 1; };
template <>          struct num_elems<__nv_fp8x2_e4m3>  { static constexpr int value = 2; };
#endif

template<typename T, int num> struct packed_as;
template<typename T>          struct packed_as<T, 1>              { using type = T; };
template<>                    struct packed_as<half,  2>          { using type = half2; };
template<>                    struct packed_as<float,  2>         { using type = float2; };
template<>                    struct packed_as<int8_t, 2>         { using type = int16_t; };
template<>                    struct packed_as<int32_t, 2>        { using type = int2; };
template<>                    struct packed_as<half2, 1>          { using type = half; };
template<>                    struct packed_as<float2, 1>         { using type = float; };
#ifdef ENABLE_BF16
template<> struct packed_as<__nv_bfloat16,  2> { using type = __nv_bfloat162; };
template<> struct packed_as<__nv_bfloat162, 1> { using type = __nv_bfloat16;  };
#endif
#ifdef ENABLE_FP8
template<> struct packed_as<__nv_fp8_e4m3,  2> { using type = __nv_fp8x2_e4m3; };
template<> struct packed_as<__nv_fp8x2_e4m3, 1> { using type = __nv_fp8_e4m3;  };
template<> struct packed_as<__nv_fp8_e5m2,  2> { using type = __nv_fp8x2_e5m2; };
template<> struct packed_as<__nv_fp8x2_e5m2, 1> { using type = __nv_fp8_e5m2;  };
#endif

inline __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __device__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __device__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }

inline __device__ float2 operator*(float2 a, float  b) { return make_float2(a.x * b, a.y * b); }
inline __device__ float2 operator+(float2 a, float  b) { return make_float2(a.x + b, a.y + b); }
inline __device__ float2 operator-(float2 a, float  b) { return make_float2(a.x - b, a.y - b); }

// clang-format on

template <typename T>
struct CudaDataType {
};

template <>
struct CudaDataType<float> {
  static constexpr cudaDataType_t value = cudaDataType::CUDA_R_32F;
};

template <>
struct CudaDataType<half> {
  static constexpr cudaDataType_t value = cudaDataType::CUDA_R_16F;
};

#ifdef ENABLE_BF16
template <>
struct CudaDataType<__nv_bfloat16> {
  static constexpr cudaDataType_t value = cudaDataType::CUDA_R_16BF;
};
#endif

inline int getSMVersion() {
  int device{-1};
  check_cuda_error(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  check_cuda_error(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  check_cuda_error(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

inline int getDevice() {
  int deviceID{0};
  check_cuda_error(cudaGetDevice(&deviceID));
  return deviceID;
}

inline int getDeviceCount() {
  int count{0};
  check_cuda_error(cudaGetDeviceCount(&count));
  return count;
}

/// @brief Identifies the memory type of the given pointer.
template <typename T>
cudaMemoryType getPtrCudaMemoryType(T* ptr) {
  cudaPointerAttributes attributes{};
  check_cuda_error(cudaPointerGetAttributes(&attributes, ptr));
  return attributes.type;
}

/// Get the memory info
/// \return The free and total amount of memory in bytes
inline std::tuple<size_t, size_t> getDeviceMemoryInfo(bool const useUvm) {
  if (useUvm) {
    size_t freeSysMem = 0;
    size_t totalSysMem = 0;
#ifndef _WIN32  // Linux
    struct sysinfo info{};

    sysinfo(&info);
    totalSysMem = info.totalram * info.mem_unit;
    freeSysMem = info.freeram * info.mem_unit;
#else   // Windows
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(memInfo);
    GlobalMemoryStatusEx(&memInfo);
    totalSysMem = memInfo.ullTotalPhys;
    freeSysMem = memInfo.ullAvailPhys;
#endif  // WIN32

    TLLM_LOG_INFO("Using UVM based system memory for KV cache, total memory %0.2f GB, available memory %0.2f GB",
                  ((double)totalSysMem / 1e9), ((double)freeSysMem / 1e9));
    return {freeSysMem, totalSysMem};
  }

  size_t free = 0;
  size_t total = 0;
  check_cuda_error(cudaMemGetInfo(&free, &total));
  TLLM_LOG_DEBUG("Using GPU memory for KV cache, total memory %0.2f GB, available memory %0.2f GB",
                 ((double)total / 1e9), ((double)free / 1e9));
  return {free, total};
}

/// @brief Gets the memory allocation granularity for the current device.
///
/// @return size_t The size of the smallest difference in memory size supported by the current device.
inline size_t getAllocationGranularity() {
  auto const currentDevice = getDevice();
  ::CUmemAllocationProp prop = {};

  prop.type = ::CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = ::CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = currentDevice;
  prop.requestedHandleTypes = ::CU_MEM_HANDLE_TYPE_NONE;

  // Get the minimum granularity supported for allocation with cuMemCreate()
  size_t granularity = 0;
  TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return granularity;
}

inline int getMultiProcessorCount() {
  int nSM{0};
  int deviceID{0};
  check_cuda_error(cudaGetDevice(&deviceID));
  check_cuda_error(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID));
  return nSM;
}

inline int getMaxSharedMemoryPerSM() {
  int nByteMaxSharedMemoryPerSM{0};
  int deviceID{0};
  check_cuda_error(cudaGetDevice(&deviceID));
  check_cuda_error(
      cudaDeviceGetAttribute(&nByteMaxSharedMemoryPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, deviceID));
  return nByteMaxSharedMemoryPerSM;
}

inline int getMaxSharedMemoryPerBlockOptin() {
  int nByteMaxSharedMemoryPerBlockOptin{0};
  int deviceID{0};
  check_cuda_error(cudaGetDevice(&deviceID));
  check_cuda_error(
      cudaDeviceGetAttribute(&nByteMaxSharedMemoryPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID));
  return nByteMaxSharedMemoryPerBlockOptin;
}

template <typename T1, typename T2>
inline size_t divUp(T1 const& a, T2 const& b) {
  auto const tmp_a = static_cast<size_t>(a);
  auto const tmp_b = static_cast<size_t>(b);
  return (tmp_a + tmp_b - 1) / tmp_b;
}

inline int roundUp(int a, int b) {
  return divUp(a, b) * b;
}

template <typename T, typename U, typename = std::enable_if_t<std::is_integral<T>::value>,
          typename = std::enable_if_t<std::is_integral<U>::value>>
auto constexpr ceilDiv(T numerator, U denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename T>
void printArrayInfo(T const* ptr, uint64_t nElement = 1, std::string name = "", bool const bPrintElement = false) {
  if (ptr == nullptr) {
    TLLM_LOG_WARNING("%s is an nullptr, skip!", name.c_str());
    return;
  }
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());

  bool const isDevicePtr = (getPtrCudaMemoryType(ptr) == cudaMemoryTypeDevice);
  size_t sizeInByte = sizeof(T) * nElement;
  TLLM_LOG_TRACE("addr=%p, location=%s, sizeof(T)=%lu, nElement=%d, sizeInByte=%lu\n", ptr,
                 (isDevicePtr ? "Device" : "Host"), sizeof(T), nElement, sizeInByte);
  T* tmp = const_cast<T*>(ptr);
  std::vector<T> tmpVec;  // For device pointer
  if (isDevicePtr) {
    tmpVec.resize(nElement);
    tmp = tmpVec.data();
    check_cuda_error(cudaMemcpy(tmp, ptr, sizeInByte, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
  }

  size_t nInf = 0;
  size_t nNaN = 0;
  size_t nZero = 0;
  double sum = 0.0;
  double sqrSum = 0.0;
  double absSum = 0.0;
  float allMax = -1.0e6f;
  float allMin = 1.0e6f;
  float allSad = 0.0f;  // Sum Abs of Difference, to distinguish A and its transpose
  float old = 0.0f;
  for (uint64_t i = 0; i < nElement; i++) {
    float val = (float)tmp[i];

    if (std::isinf(val)) {
      nInf++;
      continue;
    }
    if (std::isnan(val)) {
      nNaN++;
      continue;
    }
    nZero += (val == 0.0f);
    sum += val;
    sqrSum += val * val;
    absSum += expf(val);
    allMax = std::max(allMax, val);
    allMin = std::min(allMin, val);
    allSad += abs(val - old);
    old = val;
  }
  float avg = sum / nElement;
  float std = sqrtf(sqrSum / nElement - avg * avg);

  TLLM_LOG_INFO("%s", name.c_str());
  TLLM_LOG_INFO("size=%u, nInf=%zu, nNaN=%zu, nZero=%zu", nElement, nInf, nNaN, nZero);
  TLLM_LOG_INFO("avg=%f, absSum: %f, std=%f, max=%f, min=%f, sad=%f", avg, absSum, std, allMax, allMin, allSad);

  if (bPrintElement) {
    uint64_t constexpr nHead = 5;
    std::stringstream ss;
    ss << std::setw(10) << std::fixed << std::setprecision(3);
    for (uint64_t i = 0; i < std::min(nElement, nHead); ++i) {
      ss << (float)tmp[i] << ", ";
    }
    if (nElement > nHead) {
      ss << " ... ";
      for (uint64_t i = nElement - nHead; i < nElement; ++i) {
        ss << (float)tmp[i] << ", ";
      }
    }
    TLLM_LOG_INFO("%s", ss.str().c_str());
  }
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
}

template void printArrayInfo(float const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
template void printArrayInfo(half const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
#ifdef ENABLE_BF16
template void printArrayInfo(__nv_bfloat16 const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
#endif
#ifdef ENABLE_FP8
template void printArrayInfo(__nv_fp8_e4m3 const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
#endif
template void printArrayInfo(uint32_t const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
template void printArrayInfo(uint64_t const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
template void printArrayInfo(int const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);
template void printArrayInfo(uint8_t const* ptr, uint64_t nElement, std::string name, bool const bPrintElement);

template <typename T>
void printToStream(T const* ptr, int const nElement, FILE* strm) {
  bool const split_rows = (strm == stdout);
  if (ptr == nullptr) {
    TLLM_LOG_WARNING("Nullptr, skip!\n");
    return;
  }
  std::vector<T> tmp(nElement, 0);
  check_cuda_error(cudaMemcpy(tmp.data(), ptr, sizeof(T) * nElement, cudaMemcpyDeviceToHost));
  for (int i = 0; i < nElement; ++i) {
    fprintf(strm, "%f, ", static_cast<float>(tmp[i]));
    if (split_rows && ((i + 1) % 10) == 0)
      fprintf(strm, "\n");
  }
  if (!split_rows || (nElement % 10) != 0) {
    fprintf(strm, "\n");
  }
}

template <typename T>
void printToScreen(T const* ptr, int const nElement) {
  printToStream(ptr, nElement, stdout);
}

template <typename T>
void print2dToStream(T const* ptr, int const nRow, int const nCol, int const nStride, FILE* strm) {
  if (ptr == nullptr) {
    TLLM_LOG_WARNING("Nullptr, skip!\n");
    return;
  }
  for (int ri = 0; ri < nRow; ++ri) {
    T const* tmp = ptr + ri * nStride;
    printToStream(tmp, nCol, strm);
  }
  fprintf(strm, "\n");
}

template <typename T>
void print2dToScreen(T const* ptr, int const nRow, int const nCol, int const nStride) {
  print2dToStream(ptr, nRow, nCol, nStride, stdout);
}

template <typename T>
void print2dToFile(std::string fname, T const* ptr, int const nRow, int const nCol, int const nStride) {
  FILE* fp = fopen(fname.c_str(), "wt");
  if (fp != nullptr) {
    print2dToStream(ptr, nRow, nCol, nStride, fp);
    fclose(fp);
  }
}

__host__ __device__ inline void print_float_(float x) {
  printf("%7.3f ", x);
}

__host__ __device__ inline void print_element_(float x) {
  print_float_(x);
}

__host__ __device__ inline void print_element_(half x) {
  print_float_((float)x);
}

#ifdef ENABLE_BF16
__host__ __device__ inline void print_element_(__nv_bfloat16 x) {
  print_float_((float)x);
}
#endif

#ifdef ENABLE_FP8
__host__ __device__ inline void print_element_(__nv_fp8_e4m3 x) {
  print_float_((float)x);
}
#endif

__host__ __device__ inline void print_element_(uint8_t ui) {
  printf("%7" PRIu32 " ", (unsigned int)ui);
}

__host__ __device__ inline void print_element_(uint32_t ul) {
  printf("%7" PRIu32 " ", ul);
}

__host__ __device__ inline void print_element_(uint64_t ull) {
  printf("%7" PRIu64 " ", ull);
}

__host__ __device__ inline void print_element_(int32_t il) {
  printf("%7" PRId32 " ", il);
}

__host__ __device__ inline void print_element_(int64_t ill) {
  printf("%7" PRId64 " ", ill);
}

template <typename T>
__host__ __device__ inline void print_elements(T const* ptr, int nRow, int nCol, int nStride) {
  for (int iRow = -1; iRow < nRow; ++iRow) {
    if (iRow >= 0) {
      printf("%07d|", iRow);
    } else {
      printf("       |");  // heading row
    }
    for (int iCol = 0; iCol < nCol; iCol += 1) {
      if (iRow >= 0) {
        print_element_(ptr[iRow * nStride + iCol]);
      } else {
        printf("%7d|", iCol);  // heading colume
      }
    }
    printf("\n");
  }
}

template <typename T>
inline void printMatrix(T const* ptr, int nRow, int nCol, int nStride) {
  // `nRow` is length of row dimension
  // `nStride` is length of column dimension
  // `nCol` (<= nStride) is length for print per row
  if (ptr == nullptr) {
    TLLM_LOG_WARNING("Nullptr, skip!\n");
    return;
  }
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());

  bool const isDevicePtr = (getPtrCudaMemoryType(ptr) == cudaMemoryTypeDevice);
  size_t sizeInByte = sizeof(T) * nRow * nStride;
  TLLM_LOG_TRACE("addr=%p, location=%s, sizeof(T)=%lu, nRow=%d, nStride=%d, sizeInByte=%lu\n", ptr,
                 (isDevicePtr ? "Device" : "Host"), sizeof(T), nRow, nStride, sizeInByte);
  if (isDevicePtr) {
    std::vector<T> tmpVec;
    tmpVec.resize(nRow * nStride);
    T* tmp = tmpVec.data();
    check_cuda_error(cudaMemcpy(tmp, ptr, sizeInByte, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    print_elements(tmp, nRow, nCol, nStride);
  } else {
    print_elements(ptr, nRow, nCol, nStride);
  }
}

template void printMatrix(float const* ptr, int nRow, int nCol, int nStride);
template void printMatrix(half const* ptr, int nRow, int nCol, int nStride);
#ifdef ENABLE_BF16
template void printMatrix(__nv_bfloat16 const* ptr, int nRow, int nCol, int nStride);
#endif
#ifdef ENABLE_FP8
template void printMatrix(__nv_fp8_e4m3 const* ptr, int nRow, int nCol, int nStride);
#endif
template void printMatrix(uint32_t const* ptr, int nRow, int nCol, int nStride);
template void printMatrix(uint64_t const* ptr, int nRow, int nCol, int nStride);
template void printMatrix(int const* ptr, int nRow, int nCol, int nStride);
template void printMatrix(uint8_t const* ptr, int nRow, int nCol, int nStride);

template <typename T>
__device__ inline void printMatrixDevice(T const* ptr, int nRow, int nCol, int nStride) {
  // `nRow` is length of row dimension
  // `nStride` is length of column dimension
  // `nCol` (<= nStride) is length for print per row
  // Can be called inside kernels by one single thread
  if (ptr == nullptr) {
    printf("Nullptr, skip!\n");
    return;
  }
  size_t sizeInByte = sizeof(T) * nRow * nStride;
  printf("addr=%p, sizeof(T)=%lu, nRow=%d, nStride=%d, sizeInByte=%lu\n", ptr, sizeof(T), nRow, nStride, sizeInByte);
  print_elements(ptr, nRow, nCol, nStride);
}

template __device__ void printMatrixDevice(float const* ptr, int nRow, int nCol, int nStride);
template __device__ void printMatrixDevice(half const* ptr, int nRow, int nCol, int nStride);
#ifdef ENABLE_BF16
template __device__ void printMatrixDevice(__nv_bfloat16 const* ptr, int nRow, int nCol, int nStride);
#endif
#ifdef ENABLE_FP8
template __device__ void printMatrixDevice(__nv_fp8_e4m3 const* ptr, int nRow, int nCol, int nStride);
#endif
template __device__ void printMatrixDevice(uint32_t const* ptr, int nRow, int nCol, int nStride);
template __device__ void printMatrixDevice(uint64_t const* ptr, int nRow, int nCol, int nStride);
template __device__ void printMatrixDevice(int const* ptr, int nRow, int nCol, int nStride);
template __device__ void printMatrixDevice(uint8_t const* ptr, int nRow, int nCol, int nStride);

}  // namespace onnxruntime::llm::common

/*
 * Macros compliant with TensorRT coding conventions
 */
#define TLLM_CUDA_CHECK(stat)                                           \
  do {                                                                  \
    onnxruntime::llm::common::check((stat), #stat, __FILE__, __LINE__); \
  } while (0)

// We use singleton memory pool and the order of destructors depends on the compiler implementation. We find that the
// cudaFree/cudaFreeHost is called after cudaruntime destruction on Windows. There will be an cudaErrorCudartUnloading
// error.  However, it is safe to ignore this error because the cuda runtime is already exited, we are no more worried
// about the memory leaks.
#define TLLM_CUDA_CHECK_FREE_RESOURCE(stat)                                                                        \
  do {                                                                                                             \
    onnxruntime::llm::common::checkEx((stat), {cudaSuccess, cudaErrorCudartUnloading}, #stat, __FILE__, __LINE__); \
  } while (0)
