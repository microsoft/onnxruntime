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
#include "contrib_ops/cuda/llm/common/cudaFp8Utils.h"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

#include <algorithm>
#include <cassert>
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
#ifndef _WIN32 // Linux
#include <sys/sysinfo.h>
#endif         // not WIN32
#include <vector>
#ifdef _WIN32  // Windows
#include <windows.h>
#undef ERROR   // A Windows header file defines ERROR as 0, but it's used in our logger.h enum. Logging breaks without
               // this undef.
#endif         // WIN32


#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif



namespace ort_llm::common
{

// workspace for cublas gemm : 32MB
#define CUBLAS_WORKSPACE_SIZE 33554432

typedef struct __align__(4)
{
    half x, y, z, w;
}

half4;


inline std::optional<bool> isCudaLaunchBlocking()
{
    thread_local bool firstCall = true;
    thread_local std::optional<bool> result = std::nullopt;
    if (!firstCall)
    {
        char const* env = std::getenv("CUDA_LAUNCH_BLOCKING");
        if (env != nullptr && std::string(env) == "1")
        {
            result = true;
        }
        else
        {
            result = false;
        }
        firstCall = false;
    }
    return result;
}

inline bool isCapturing(cudaStream_t stream)
{
    cudaStreamCaptureStatus status;
    CUDA_CALL_THROW(cudaStreamIsCapturing(stream, &status));
    return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive;
}

inline bool doCheckError(cudaStream_t stream)
{
    auto const cudaLaunchBlocking = isCudaLaunchBlocking();
    if (cudaLaunchBlocking.has_value() && cudaLaunchBlocking.value())
    {
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

inline void syncAndCheck(cudaStream_t stream, char const* const file, int const line)
{
    if (doCheckError(stream))
    {
        cudaStreamSynchronize(stream);
        CUDA_CALL_THROW(cudaGetLastError());
    }
}

#define PRINT_FUNC_NAME_()                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        std::cout << "[OnnxRuntime-LLM][CALL] " << __FUNCTION__ << " " << std::endl;                                      \
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
struct CudaDataType
{
};

template <>
struct CudaDataType<float>
{
    static constexpr cudaDataType_t value = cudaDataType::CUDA_R_32F;
};

template <>
struct CudaDataType<half>
{
    static constexpr cudaDataType_t value = cudaDataType::CUDA_R_16F;
};

#ifdef ENABLE_BF16
template <>
struct CudaDataType<__nv_bfloat16>
{
    static constexpr cudaDataType_t value = cudaDataType::CUDA_R_16BF;
};
#endif

inline int getSMVersion()
{
    int device{-1};
    CUDA_CALL_THROW(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    CUDA_CALL_THROW(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CALL_THROW(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

inline int getDevice()
{
    int deviceID{0};
    CUDA_CALL_THROW(cudaGetDevice(&deviceID));
    return deviceID;
}

inline int getDeviceCount()
{
    int count{0};
    CUDA_CALL_THROW(cudaGetDeviceCount(&count));
    return count;
}

/// @brief Identifies the memory type of the given pointer.
template <typename T>
cudaMemoryType getPtrCudaMemoryType(T* ptr)
{
    cudaPointerAttributes attributes{};
    CUDA_CALL_THROW(cudaPointerGetAttributes(&attributes, ptr));
    return attributes.type;
}

/// Get the memory info
/// \return The free and total amount of memory in bytes
inline std::tuple<size_t, size_t> getDeviceMemoryInfo(bool const useUvm)
{
    if (useUvm)
    {
        size_t freeSysMem = 0;
        size_t totalSysMem = 0;
#ifndef _WIN32 // Linux
        struct sysinfo info
        {
        };

        sysinfo(&info);
        totalSysMem = info.totalram * info.mem_unit;
        freeSysMem = info.freeram * info.mem_unit;
#else  // Windows
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(memInfo);
        GlobalMemoryStatusEx(&memInfo);
        totalSysMem = memInfo.ullTotalPhys;
        freeSysMem = memInfo.ullAvailPhys;
#endif // WIN32

        ORT_LLM_LOG_INFO("Using UVM based system memory for KV cache, total memory %0.2f GB, available memory %0.2f GB",
            ((double) totalSysMem / 1e9), ((double) freeSysMem / 1e9));
        return {freeSysMem, totalSysMem};
    }

    size_t free = 0;
    size_t total = 0;
    CUDA_CALL_THROW(cudaMemGetInfo(&free, &total));
    ORT_LLM_LOG_DEBUG("Using GPU memory for KV cache, total memory %0.2f GB, available memory %0.2f GB",
        ((double) total / 1e9), ((double) free / 1e9));
    return {free, total};
}

inline int getMultiProcessorCount()
{
    int nSM{0};
    int deviceID{0};
    CUDA_CALL_THROW(cudaGetDevice(&deviceID));
    CUDA_CALL_THROW(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID));
    return nSM;
}

inline int getMaxSharedMemoryPerSM()
{
    int nByteMaxSharedMemoryPerSM{0};
    int deviceID{0};
    CUDA_CALL_THROW(cudaGetDevice(&deviceID));
    CUDA_CALL_THROW(
        cudaDeviceGetAttribute(&nByteMaxSharedMemoryPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, deviceID));
    return nByteMaxSharedMemoryPerSM;
}

inline int getMaxSharedMemoryPerBlockOptin()
{
    int nByteMaxSharedMemoryPerBlockOptin{0};
    int deviceID{0};
    CUDA_CALL_THROW(cudaGetDevice(&deviceID));
    CUDA_CALL_THROW(
        cudaDeviceGetAttribute(&nByteMaxSharedMemoryPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID));
    return nByteMaxSharedMemoryPerBlockOptin;
}

template <typename T1, typename T2>
inline size_t divUp(T1 const& a, T2 const& b)
{
    auto const tmp_a = static_cast<size_t>(a);
    auto const tmp_b = static_cast<size_t>(b);
    return (tmp_a + tmp_b - 1) / tmp_b;
}

inline int roundUp(int a, int b)
{
    return divUp(a, b) * b;
}

template <typename T, typename U, typename = std::enable_if_t<std::is_integral<T>::value>,
    typename = std::enable_if_t<std::is_integral<U>::value>>
auto constexpr ceilDiv(T numerator, U denominator)
{
    return (numerator + denominator - 1) / denominator;
}

template <typename T>
void printArrayInfo(T const* ptr, uint64_t nElement = 1, std::string name = "", bool const bPrintElement = false)
{
    if (ptr == nullptr)
    {
        ORT_LLM_LOG_WARNING("%s is an nullptr, skip!", name.c_str());
        return;
    }
    cudaDeviceSynchronize();
    CUDA_CALL_THROW(cudaGetLastError());

    bool const isDevicePtr = (getPtrCudaMemoryType(ptr) == cudaMemoryTypeDevice);
    size_t sizeInByte = sizeof(T) * nElement;
    ORT_LLM_LOG_TRACE("addr=%p, location=%s, sizeof(T)=%lu, nElement=%d, sizeInByte=%lu\n", ptr,
        (isDevicePtr ? "Device" : "Host"), sizeof(T), nElement, sizeInByte);
    T* tmp = const_cast<T*>(ptr);
    std::vector<T> tmpVec; // For device pointer
    if (isDevicePtr)
    {
        tmpVec.resize(nElement);
        tmp = tmpVec.data();
        CUDA_CALL_THROW(cudaMemcpy(tmp, ptr, sizeInByte, cudaMemcpyDeviceToHost));
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
    float allSad = 0.0f; // Sum Abs of Difference, to distinguish A and its transpose
    float old = 0.0f;
    for (uint64_t i = 0; i < nElement; i++)
    {
        float val = (float) tmp[i];

        if (std::isinf(val))
        {
            nInf++;
            continue;
        }
        if (std::isnan(val))
        {
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

    ORT_LLM_LOG_INFO("%s", name.c_str());
    ORT_LLM_LOG_INFO("size=%u, nInf=%zu, nNaN=%zu, nZero=%zu", nElement, nInf, nNaN, nZero);
    ORT_LLM_LOG_INFO("avg=%f, absSum: %f, std=%f, max=%f, min=%f, sad=%f", avg, absSum, std, allMax, allMin, allSad);

    if (bPrintElement)
    {
        uint64_t constexpr nHead = 5;
        std::stringstream ss;
        ss << std::setw(10) << std::fixed << std::setprecision(3);
        for (uint64_t i = 0; i < std::min(nElement, nHead); ++i)
        {
            ss << (float) tmp[i] << ", ";
        }
        if (nElement > nHead)
        {
            ss << " ... ";
            for (uint64_t i = nElement - nHead; i < nElement; ++i)
            {
                ss << (float) tmp[i] << ", ";
            }
        }
        ORT_LLM_LOG_INFO("%s", ss.str().c_str());
    }
    cudaDeviceSynchronize();
    CUDA_CALL_THROW(cudaGetLastError());
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
void printToStream(T const* ptr, int const nElement, FILE* strm)
{
    bool const split_rows = (strm == stdout);
    if (ptr == nullptr)
    {
        ORT_LLM_LOG_WARNING("Nullptr, skip!\n");
        return;
    }
    std::vector<T> tmp(nElement, 0);
    CUDA_CALL_THROW(cudaMemcpy(tmp.data(), ptr, sizeof(T) * nElement, cudaMemcpyDeviceToHost));
    for (int i = 0; i < nElement; ++i)
    {
        fprintf(strm, "%f, ", static_cast<float>(tmp[i]));
        if (split_rows && ((i + 1) % 10) == 0)
            fprintf(strm, "\n");
    }
    if (!split_rows || (nElement % 10) != 0)
    {
        fprintf(strm, "\n");
    }
}

template <typename T>
void printToScreen(T const* ptr, int const nElement)
{
    printToStream(ptr, nElement, stdout);
}

template <typename T>
void print2dToStream(T const* ptr, int const nRow, int const nCol, int const nStride, FILE* strm)
{
    if (ptr == nullptr)
    {
        ORT_LLM_LOG_WARNING("Nullptr, skip!\n");
        return;
    }
    for (int ri = 0; ri < nRow; ++ri)
    {
        T const* tmp = ptr + ri * nStride;
        printToStream(tmp, nCol, strm);
    }
    fprintf(strm, "\n");
}

template <typename T>
void print2dToScreen(T const* ptr, int const nRow, int const nCol, int const nStride)
{
    print2dToStream(ptr, nRow, nCol, nStride, stdout);
}

template <typename T>
void print2dToFile(std::string fname, T const* ptr, int const nRow, int const nCol, int const nStride)
{
    FILE* fp = fopen(fname.c_str(), "wt");
    if (fp != nullptr)
    {
        print2dToStream(ptr, nRow, nCol, nStride, fp);
        fclose(fp);
    }
}

__host__ __device__ inline void print_float_(float x)
{
    printf("%7.3f ", x);
}

__host__ __device__ inline void print_element_(float x)
{
    print_float_(x);
}

__host__ __device__ inline void print_element_(half x)
{
    print_float_((float) x);
}

#ifdef ENABLE_BF16
__host__ __device__ inline void print_element_(__nv_bfloat16 x)
{
    print_float_((float) x);
}
#endif

#ifdef ENABLE_FP8
__host__ __device__ inline void print_element_(__nv_fp8_e4m3 x)
{
    print_float_((float) x);
}
#endif

__host__ __device__ inline void print_element_(uint8_t ui)
{
    printf("%7" PRIu32 " ", (unsigned int) ui);
}

__host__ __device__ inline void print_element_(uint32_t ul)
{
    printf("%7" PRIu32 " ", ul);
}

__host__ __device__ inline void print_element_(uint64_t ull)
{
    printf("%7" PRIu64 " ", ull);
}

__host__ __device__ inline void print_element_(int32_t il)
{
    printf("%7" PRId32 " ", il);
}

__host__ __device__ inline void print_element_(int64_t ill)
{
    printf("%7" PRId64 " ", ill);
}

template <typename T>
__host__ __device__ inline void print_elements(T const* ptr, int nRow, int nCol, int nStride)
{
    for (int iRow = -1; iRow < nRow; ++iRow)
    {
        if (iRow >= 0)
        {
            printf("%07d|", iRow);
        }
        else
        {
            printf("       |"); // heading row
        }
        for (int iCol = 0; iCol < nCol; iCol += 1)
        {
            if (iRow >= 0)
            {
                print_element_(ptr[iRow * nStride + iCol]);
            }
            else
            {
                printf("%7d|", iCol); // heading colume
            }
        }
        printf("\n");
    }
    printf("\n");
}

template <typename T>
inline void printMatrix(T const* ptr, int nRow, int nCol, int nStride)
{
    // `nRow` is length of row dimension
    // `nStride` is length of column dimension
    // `nCol` (<= nStride) is length for print per row
    if (ptr == nullptr)
    {
        ORT_LLM_LOG_WARNING("Nullptr, skip!\n");
        return;
    }
    cudaDeviceSynchronize();
    CUDA_CALL_THROW(cudaGetLastError());

    bool const isDevicePtr = (getPtrCudaMemoryType(ptr) == cudaMemoryTypeDevice);
    size_t sizeInByte = sizeof(T) * nRow * nStride;
    ORT_LLM_LOG_TRACE("addr=%p, location=%s, sizeof(T)=%lu, nRow=%d, nStride=%d, sizeInByte=%lu\n", ptr,
        (isDevicePtr ? "Device" : "Host"), sizeof(T), nRow, nStride, sizeInByte);
    if (isDevicePtr)
    {
        std::vector<T> tmpVec;
        tmpVec.resize(nRow * nStride);
        T* tmp = tmpVec.data();
        CUDA_CALL_THROW(cudaMemcpy(tmp, ptr, sizeInByte, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        CUDA_CALL_THROW(cudaGetLastError());
        print_elements(tmp, nRow, nCol, nStride);
    }
    else
    {
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
__device__ inline void printMatrixDevice(T const* ptr, int nRow, int nCol, int nStride)
{
    // `nRow` is length of row dimension
    // `nStride` is length of column dimension
    // `nCol` (<= nStride) is length for print per row
    // Can be called inside kernels by one single thread
    if (ptr == nullptr)
    {
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

#ifndef CUDA_CALL
#define CUDA_CALL(answer)                                                                                              \
    {                                                                                                                  \
        gpuAssert((answer), __FILE__, __LINE__);                                                                       \
    }

inline void gpuAssert(cudaError_t code, char const* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s @ %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

inline void gpuAssert(CUresult code, char const* file, int line, bool abort = true)
{
    if (code != CUresult::CUDA_SUCCESS)
    {
        char const* buf = "Unknown error";
        assert(cuGetErrorString(code, &buf) == CUresult::CUDA_SUCCESS);
        fprintf(stderr, "Driver API error: %s @ %s:%d\n", buf, file, line);
        if (abort)
            exit(code);
    }
}
#endif

template <typename T>
struct UpperType;

template <>
struct UpperType<int8_t>
{
    using Type = int;
};

template <>
struct UpperType<uint32_t>
{
    using Type = uint32_t;
};

template <>
struct UpperType<int>
{
    using Type = int;
};

template <>
struct UpperType<__nv_bfloat16>
{
    using Type = double;
};

template <>
struct UpperType<half>
{
    using Type = double;
};

template <>
struct UpperType<float>
{
    using Type = double;
};

extern "C"
{
    __device__ uint32_t __nvvm_get_smem_pointer(void* ptr);
}

__forceinline__ __device__ void issue_stas(uint32_t dist_barrier_ptr, uint32_t dist_buffer_ptr, uint32_t d0)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
    asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %2, [%1];\n\t"
                 :
                 : "r"(dist_buffer_ptr), "r"(dist_barrier_ptr), "r"(d0));
#endif
}

__forceinline__ __device__ void issue_stas(uint32_t dist_barrier_ptr, uint32_t dist_buffer_ptr, uint64_t d0)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
    asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %2, [%1];\n\t"
                 :
                 : "r"(dist_buffer_ptr), "r"(dist_barrier_ptr), "l"(d0));
#endif
}

__forceinline__ __device__ void issue_stas(
    uint32_t dist_barrier_ptr, uint32_t dist_buffer_ptr, uint32_t d0, uint32_t d1)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
    asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%2, %3}, [%1];\n\t"
                 :
                 : "r"(dist_buffer_ptr), "r"(dist_barrier_ptr), "r"(d0), "r"(d1));
#endif
}

__forceinline__ __device__ void issue_stas(
    uint32_t dist_barrier_ptr, uint32_t dist_buffer_ptr, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%2, %3, %4, %5}, [%1];\n\t"
                 :
                 : "r"(dist_buffer_ptr), "r"(dist_barrier_ptr), "r"(d0), "r"(d1), "r"(d2), "r"(d3));
#endif
}

inline __device__ uint32_t elect_one_sync()
{
    uint32_t pred = 0;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
#if (defined(__CUDA_ARCH_FEAT_SM90_ALL))
    uint32_t laneid = 0;
    asm volatile(
        "\n\
    {\n\
        .reg .b32 %rx;\n\
        .reg .pred %px;\n\
        elect.sync %rx|%px, %2;\n\
        @%px mov.s32 %1, 1;\n\
        mov.s32 %0, %rx;\n\
    }\n\
  "
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
#endif
#endif
    return pred;
}

__forceinline__ __device__ uint32_t get_smem_pointer(void const* ptr)
{
    return __nvvm_get_smem_pointer(const_cast<void*>(ptr));
}

__forceinline__ __device__ void bar_create(void* bar_ptr, int init_count)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    unsigned smem_ptr = get_smem_pointer(bar_ptr);

    asm volatile(
        "{\n\t"
        "mbarrier.init.shared.b64 [%1], %0; \n\t"
        "}"
        :
        : "r"(init_count), "r"(smem_ptr));
#endif
}

struct Arrive_wait
{
public:
    __forceinline__ __device__ Arrive_wait()
    {
        bar_base_ = NULL;
    }

    __forceinline__ __device__ Arrive_wait(uint64_t* bar_base, int id = 0)
    {
        bar_base_ = bar_base;
        id_ = id;
    }

    __forceinline__ __device__ int bar_peek(int id, unsigned int bar_phase)
    {
        uint32_t result32{};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
        auto* bar_ptr = bar_base_ + id;
        unsigned smem_ptr = get_smem_pointer(bar_ptr);
        asm volatile(
            "{\n\t"
            ".reg .pred       P1; \n\t"
            "mbarrier.try_wait.parity.shared.b64 P1, [%1], %2; \n\t"
            "selp.b32 %0, 1, 0, P1; \n\t"
            "}"
            : "=r"(result32)
            : "r"(smem_ptr), "r"(bar_phase));
#endif
        return result32;
    }

    __forceinline__ __device__ int bar_peek(int id, unsigned int bar_phase, int pred)
    {
        uint32_t result32{};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
        auto* bar_ptr = bar_base_ + id;
        unsigned smem_ptr = get_smem_pointer(bar_ptr);
        asm volatile(
            "{\n\t"
            ".reg .pred       P1; \n\t"
            ".reg .pred P2;\n\t"
            "setp.eq.u32 P2, %3, 1;\n\t"
            "@P2 mbarrier.try_wait.parity.shared.b64 P1, [%1], %2; \n\t"
            "selp.b32 %0, 1, 0, P1; \n\t"
            "}"
            : "=r"(result32)
            : "r"(smem_ptr), "r"(bar_phase), "r"(pred));
#endif
        return result32;
    }

    __forceinline__ __device__ void bar_wait(int id, unsigned int bar_phase)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
        auto* bar_ptr = bar_base_ + id;
        unsigned smem_ptr = get_smem_pointer(bar_ptr);
        asm volatile(
            "{\n\t"
            ".reg .pred                P1; \n\t"
            "LAB_WAIT: \n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1; \n\t"
            "@P1                       bra.uni DONE; \n\t"
            "bra.uni                   LAB_WAIT; \n\t"
            "DONE: \n\t"
            "}"
            :
            : "r"(smem_ptr), "r"(bar_phase));
#endif
    }

    __forceinline__ __device__ void bar_arrive_dsmem(int const& id)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
        auto* bar_ptr = bar_base_ + id;
        asm volatile(
            "{\n\t"
            "mbarrier.arrive.b64   _, [%0];\n\t"
            "}"
            :
            : "l"(bar_ptr));
#endif
    }

    __forceinline__ __device__ void bar_arrive_dsmem(int const& id, uint32_t const& pred)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
        asm volatile(
            "{\n\t"
            " .reg .pred p;\n\t"
            " .reg .s64 addr;\n\t"
            " .reg .b64 tmp;\n\t"
            "   setp.eq.u32 p, %2, 1;\n\t"
            "   mul.wide.s32 tmp, %0, 8;\n\t"
            "   add.s64 addr, tmp, %1;\n\t"
            "@p mbarrier.arrive.b64   _, [addr];\n\t"
            "}"
            :
            : "r"(id), "l"(bar_base_), "r"(pred));
#endif
    }

    // Sets up the base address for arrival with the correct ctaid in cga
    __forceinline__ __device__ void set_bar_base_dsmem(uint32_t const& cta_id)
    {
        bar_base_ = reinterpret_cast<uint64_t*>(
            (reinterpret_cast<uintptr_t>(bar_base_) & 0xFFFFFFFFF0FFFFFFULL) + (cta_id << 24));
    }

    __forceinline__ __device__ void bar_arrive_normal(int id, bool flag = true)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
        if (flag == true)
        {
            uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(bar_base_ + id);
            unsigned smem_ptr = get_smem_pointer(bar_ptr);
            asm volatile(
                "{\n\t"
                ".reg .b64 state; \n\t"
                "mbarrier.arrive.shared.b64   state, [%0];\n\t"
                "}"
                :
                : "r"(smem_ptr));
        }
#endif
    }

    __forceinline__ __device__ void bar_arrive_set_transactioncnt(int id, int expected_copy_bytes)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
        auto* bar_ptr = bar_base_ + id;
        unsigned smem_ptr = get_smem_pointer(bar_ptr);
        asm volatile(
            "{\n\t"
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1; \n\t"
            "}"
            :
            : "r"(smem_ptr), "r"(expected_copy_bytes));
#endif
    }

    __forceinline__ __device__ void bar_arrive_set_transactioncnt(int id, int expected_copy_bytes, uint32_t pred)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
        auto* bar_ptr = bar_base_ + id;
        unsigned smem_ptr = get_smem_pointer(bar_ptr);
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.eq.u32 p, %2, 1;\n\t"
            "@p mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1; \n\t"
            "}"
            :
            : "r"(smem_ptr), "r"(expected_copy_bytes), "r"(pred));
#endif
    }

    __forceinline__ __device__ uint64_t* bar_base()
    {
        return bar_base_;
    }

    __forceinline__ __device__ uint64_t* get_bar_addr(int id)
    {
        return bar_base_ + id;
    }

private:
    // smem barrier base pointer
    uint64_t* bar_base_;
    // barrier id
    int id_;
};

__forceinline__ __device__ void cga_sync()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.sync;\n" : :);
#endif
}

__forceinline__ __device__ void cga_arrive()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.arrive.aligned;\n" : :);
#endif
}

__forceinline__ __device__ void cga_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.wait.aligned;\n" : :);
#endif
}

inline __device__ void fence_view_async_shared()
{

    // only compiles on sm90+

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;\n" : :);
#endif
}

template <typename T>
__forceinline__ __device__ T* get_DSMEM_ptr(T* localAddress, uint32_t destCtaId)
{
    T* dsmemAddress
        = reinterpret_cast<T*>(((unsigned long long int) localAddress & 0xFFFFFFFFF0FFFFFFULL) + (destCtaId << 24));
    return dsmemAddress;
}

template <typename T>
__forceinline__ __device__ void write_DSMEM_Address(T* localAddress, uint32_t destCtaId, T val)
{
    T* dsmemAddress = get_DSMEM_ptr(localAddress, destCtaId);
    *dsmemAddress = val;
}

__forceinline__ __device__ void arrive_barrier(uint64_t* p_barrier, uint32_t arrive_cnt = 1)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    asm volatile("{mbarrier.arrive.shared.b64   _, [%0],%1;\n\t}" : : "l"(p_barrier), "r"(arrive_cnt));
#endif
}

__forceinline__ __device__ void arrive_DSMEM_barrier(uint64_t* p_barrier, uint32_t ctaid)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    uint64_t* p_barrier_remote = get_DSMEM_ptr(p_barrier, ctaid);
    asm volatile("{mbarrier.arrive.b64   _, [%0];\n\t}" : : "l"(p_barrier_remote));
#endif
}

__forceinline__ __device__ void arrive_DSMEM_barrier_and_set_tx_cnt(
    uint64_t* p_barrier, uint32_t ctaid, uint32_t expected_copy_bytes)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    uint32_t p_bar = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(get_DSMEM_ptr(p_barrier, ctaid)));
    asm volatile("{mbarrier.arrive.expect_tx.b64 _, [%0], %1; \n\t}" ::"r"(p_bar), "r"(expected_copy_bytes));
#endif
}

template <bool barSetTxCnt = true>
__forceinline__ __device__ void stas(uint32_t* p_data, uint64_t* p_barrier, uint32_t ctaid, uint32_t const& wrdat)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    if (barSetTxCnt)
        arrive_DSMEM_barrier_and_set_tx_cnt(p_barrier, ctaid, sizeof(uint32_t));
    uint32_t buffer_ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p_data));
    uint32_t barrier_ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p_barrier));

    uint32_t buffer_ptr_, barrier_ptr_;
    asm volatile(
        "{\n\t"
        "setctarank.shared.u32 %0, %2, %4;\n\t"
        "setctarank.shared.u32 %1, %3, %4;\n\t"
        "}"
        : "=r"(buffer_ptr_), "=r"(barrier_ptr_)
        : "r"(buffer_ptr), "r"(barrier_ptr), "r"(ctaid));
    issue_stas(buffer_ptr_, barrier_ptr_, wrdat);
#endif
}

template <bool barSetTxCnt = true>
__forceinline__ __device__ void stas(uint64_t* p_data, uint64_t* p_barrier, uint32_t ctaid, uint64_t const& wrdat)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    if (barSetTxCnt)
        arrive_DSMEM_barrier_and_set_tx_cnt(p_barrier, ctaid, sizeof(uint64_t));
    uint32_t buffer_ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p_data));
    uint32_t barrier_ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p_barrier));

    uint32_t buffer_ptr_, barrier_ptr_;
    asm volatile(
        "{\n\t"
        "setctarank.shared.u32 %0, %2, %4;\n\t"
        "setctarank.shared.u32 %1, %3, %4;\n\t"
        "}"
        : "=r"(buffer_ptr_), "=r"(barrier_ptr_)
        : "r"(buffer_ptr), "r"(barrier_ptr), "r"(ctaid));
    issue_stas(buffer_ptr_, barrier_ptr_, wrdat);
#endif
}

template <bool barSetTxCnt = true>
__forceinline__ __device__ void stas(uint64_t* p_data, uint64_t* p_barrier, uint32_t ctaid, const uint32_t wrdat0,
    const uint32_t wrdat1, const uint32_t wrdat2, const uint32_t wrdat3)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    if (barSetTxCnt)
        arrive_DSMEM_barrier_and_set_tx_cnt(p_barrier, ctaid, 4 * sizeof(uint32_t));
    uint32_t buffer_ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p_data));
    uint32_t barrier_ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p_barrier));

    uint32_t buffer_ptr_, barrier_ptr_;
    asm volatile(
        "{\n\t"
        "setctarank.shared.u32 %0, %2, %4;\n\t"
        "setctarank.shared.u32 %1, %3, %4;\n\t"
        "}"
        : "=r"(buffer_ptr_), "=r"(barrier_ptr_)
        : "r"(buffer_ptr), "r"(barrier_ptr), "r"(ctaid));
    issue_stas(buffer_ptr_, barrier_ptr_, wrdat0, wrdat1, wrdat2, wrdat3);
#endif
}

template <bool barSetTxCnt = true, bool assumeAligned = true, typename T = void>
__forceinline__ __device__ void stas(T* p_data, uint64_t* p_barrier, uint32_t ctaid, T const& wrdat)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    static_assert(sizeof(T) % 4 == 0);
    if (barSetTxCnt)
        arrive_DSMEM_barrier_and_set_tx_cnt(p_barrier, ctaid, sizeof(T));

    uint32_t buffer_ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p_data));
    uint32_t barrier_ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p_barrier));

    uint32_t buffer_ptr_, barrier_ptr_;
    asm volatile(
        "{\n\t"
        "setctarank.shared.u32 %0, %2, %4;\n\t"
        "setctarank.shared.u32 %1, %3, %4;\n\t"
        "}"
        : "=r"(buffer_ptr_), "=r"(barrier_ptr_)
        : "r"(buffer_ptr), "r"(barrier_ptr), "r"(ctaid));

    uint32_t const* p_wrdat_b32 = reinterpret_cast<uint32_t const*>(&wrdat);

    for (uint32_t offset = 0; offset < sizeof(T);)
    {
        if constexpr (assumeAligned)
        {
            if (offset + 16 <= sizeof(T))
            {
                // Use write_async_v4_b32
                issue_stas(buffer_ptr_ + offset, barrier_ptr_, p_wrdat_b32[offset / 4], p_wrdat_b32[offset / 4 + 1],
                    p_wrdat_b32[offset / 4 + 2], p_wrdat_b32[offset / 4 + 3]);
                offset += 16;
            }
            else if (offset + 8 <= sizeof(T) && (buffer_ptr + offset) % 8 == 0)
            {
                // Use write_async_v2_b32
                issue_stas(buffer_ptr + offset, barrier_ptr_, p_wrdat_b32[offset / 4], p_wrdat_b32[offset / 4 + 1]);
                offset += 8;
            }
            else
            {
                issue_stas(buffer_ptr + offset, barrier_ptr_, p_wrdat_b32[offset / 4]);
                offset += 4;
            }
        }
        else
        {
            issue_stas(buffer_ptr + offset, barrier_ptr_, p_wrdat_b32[offset / 4]);
            offset += 4;
        }
    }
#endif
}

struct OrderedMutex
{
    uint64_t barriers[2];

    __device__ void init(int tid0, int threads0, int threads1)
    {
        if (tid0)
        {
            bar_create(&barriers[0], threads0);
            bar_create(&barriers[1], threads1);
        }
    }

    OrderedMutex() = default;
    OrderedMutex(OrderedMutex const& other) = delete;
};

class OrderedMutexAccessor
{
public:
    struct State
    {
        int phase = 0;
    };

private:
    int _phase;
    int _id;
    Arrive_wait _barriers;

public:
    __device__ OrderedMutexAccessor(OrderedMutex& m, int id, State state)
        : _phase(state.phase)
        , _id(id)
        , _barriers(m.barriers)
    {
    }

    __device__ void arrive()
    {
        _barriers.bar_arrive_normal(_id);
    }

    __device__ void wait()
    {
        _barriers.bar_wait(_id ^ 1, _phase);
        _phase ^= 1;
    }

    __device__ State exportState()
    {
        return {.phase = _phase};
    }
};

template <typename T, T VALUE>
struct ConstExprWrapper
{
    static constexpr T value = VALUE;
};

template <int VALUE>
using Int = ConstExprWrapper<int, VALUE>;

template <bool VALUE>
using Bool = ConstExprWrapper<bool, VALUE>;

template <typename T>
struct TmaDescType;

template <>
struct TmaDescType<__nv_bfloat16>
{
    static constexpr auto value = CUtensorMapDataType_enum::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
};

template <>
struct TmaDescType<float>
{
    static constexpr auto value = CUtensorMapDataType_enum::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
};

#define DEFINE_MEMBER_CHECKER(member)                                                                                  \
    template <typename T, typename V = bool>                                                                           \
    struct has_##member : std::false_type                                                                              \
    {                                                                                                                  \
    };                                                                                                                 \
    template <typename T>                                                                                              \
    struct has_##member<T,                                                                                             \
        typename std::enable_if<!std::is_same<decltype(std::declval<T>().member), void>::value, bool>::type>           \
        : std::true_type                                                                                               \
    {                                                                                                                  \
    };

#define HAS_MEMBER(C, member) has_##member<C>::value

DEFINE_MEMBER_CHECKER(output)
DEFINE_MEMBER_CHECKER(residual)
DEFINE_MEMBER_CHECKER(bias)
DEFINE_MEMBER_CHECKER(deq)
DEFINE_MEMBER_CHECKER(qua)
DEFINE_MEMBER_CHECKER(high_preciecion_normed_output)

} // namespace ort_llm::common

#define TLLM_CUDA_CHECK(stat)                                  \
  do {                                                         \
    ort_llm::common::check((stat), #stat, __FILE__, __LINE__); \
  } while (0)

// We use singleton memory pool and the order of destructors depends on the compiler implementation. We find that the
// cudaFree/cudaFreeHost is called after cudaruntime destruction on Windows. There will be an cudaErrorCudartUnloading
// error.  However, it is safe to ignore this error because the cuda runtime is already exited, we are no more worried
// about the memory leaks.
#define TLLM_CUDA_CHECK_FREE_RESOURCE(stat)                                                               \
  do {                                                                                                    \
    ort_llm::common::checkEx((stat), {cudaSuccess, cudaErrorCudartUnloading}, #stat, __FILE__, __LINE__); \
  } while (0)

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
