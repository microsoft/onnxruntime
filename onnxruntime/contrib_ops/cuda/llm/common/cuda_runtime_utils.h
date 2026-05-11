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

#include <optional>
#include <cuda_runtime_api.h>
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime::llm::common {
inline int getDevice() {
  int deviceID{0};
  CUDA_CALL_THROW(cudaGetDevice(&deviceID));
  return deviceID;
}

inline int getSMVersion() {
  int device{-1};
  CUDA_CALL_THROW(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

inline int getMultiProcessorCount() {
  int nSM{0};
  int deviceID{0};
  CUDA_CALL_THROW(cudaGetDevice(&deviceID));
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID));
  return nSM;
}

inline int getMaxSharedMemoryPerBlockOptin() {
  int nByteMaxSharedMemoryPerBlockOptin{0};
  int deviceID{0};
  CUDA_CALL_THROW(cudaGetDevice(&deviceID));
  CUDA_CALL_THROW(
      cudaDeviceGetAttribute(&nByteMaxSharedMemoryPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID));
  return nByteMaxSharedMemoryPerBlockOptin;
}

inline std::optional<bool> isCudaLaunchBlocking() {
  thread_local bool firstCall = true;
  thread_local std::optional<bool> result = std::nullopt;
  if (!firstCall) {
    char const* env = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (env != nullptr && std::string(env) == "1") {
      result = true;
    } else {
      result = false;
    }
    firstCall = false;
  }
  return result;
}

inline bool isCapturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  CUDA_CALL_THROW(cudaStreamIsCapturing(stream, &status));
  return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive;
}

inline bool doCheckError(cudaStream_t stream) {
  auto const cudaLaunchBlocking = isCudaLaunchBlocking();
  if (cudaLaunchBlocking.has_value() && cudaLaunchBlocking.value()) {
    return !isCapturing(stream);
  }

#ifndef NDEBUG
  // Debug builds will sync when we're not capturing unless explicitly
  // disabled.
  bool const checkError = cudaLaunchBlocking.value_or(!isCapturing(stream));
#else
  bool const checkError = cudaLaunchBlocking.value_or(false);
#endif

  return checkError;
}

inline void syncAndCheck(cudaStream_t stream, char const* const file, int const line) {
  if (doCheckError(stream)) {
    cudaStreamSynchronize(stream);
    ::onnxruntime::CudaCall<cudaError, true>(cudaGetLastError(), "cudaGetLastError", "CUDA", cudaSuccess, "", file, line);
  }
}

#define sync_check_cuda_error(stream) onnxruntime::llm::common::syncAndCheck(stream, __FILE__, __LINE__)

template <typename T, typename U, typename = std::enable_if_t<std::is_integral<T>::value>,
          typename = std::enable_if_t<std::is_integral<U>::value>>
auto constexpr ceilDiv(T numerator, U denominator) {
  return (numerator + denominator - 1) / denominator;
}

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

template <typename T, T VALUE>
struct ConstExprWrapper {
  static constexpr T value = VALUE;
};

template <int VALUE>
using ConstInt = ConstExprWrapper<int, VALUE>;

template <bool VALUE>
using ConstBool = ConstExprWrapper<bool, VALUE>;

}  // namespace onnxruntime::llm::common
