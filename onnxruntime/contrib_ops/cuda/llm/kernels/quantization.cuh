/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "contrib_ops/cuda/llm/common/cuda_type_utils.cuh"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/common/quant_type_utils.cuh"
#include "contrib_ops/cuda/llm/common/reduce_kernel_utils.cuh"
// #include "tensorrt_llm/kernels/quantization.h"
#include <float.h>

using namespace onnxruntime::llm::common;

namespace onnxruntime::llm {
enum class FP4QuantizationSFLayout {
  // Block scale factors are stored in swizzled layout for cutlass FP4 kernel. Scale factor
  // blocks are organized in 512-byte blocks in global memory, with each block having 128x4 FP8 values.
  // The SF matrix dimensions are therefore padded - rows to the nearest multiple of 128 and columns to
  // the nearest multiple of 4.
  //
  // The scale factor block rows map to data block rows in an interleaved pattern:
  // For a scale factor row 'i', it maps to data block row: (i % 4) * 32 + (i / 4)
  // Column 'j' in the scale factor block corresponds to scaling the j-th block in the data tensor.
  //
  // Please refer to https://nvbugs/4165523 for more details about the swizzled layout.
  SWIZZLED,
  // Block scale factors are stored in linear layout (row-major). This is used in some trtllm-gen kernels standard.
  LINEAR
};

#define PadUpFn(X, Y) ((X + Y - 1) / (Y) * (Y))

// totalCloumn should be in SFMatrix, not activation Matrix, so no sfVecSize needed.
inline int computeFP4SwizzledLayoutSFSize(int totalRow, int totalColumn) {
  int paddedRow = PadUpFn(totalRow, 128);
  int paddedColumn = PadUpFn(totalColumn, 4);
  return paddedRow * paddedColumn;
}

inline int computeFP4LinearLayoutSFSize(int totalRow, int totalColumn) {
  return totalRow * totalColumn;
}

namespace kernels {

__global__ static void quantizedKernel(char4* dst, float4 const* src, int64_t const sizeDiv4, float const* scalePtr) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x) {
    float const scale = __ldg(scalePtr);
    char4 tmp;
    float4 const floatTmp = __ldg(src + idx);
    tmp.x = cuda_cast<int8_t>(floatTmp.x * scale);
    tmp.y = cuda_cast<int8_t>(floatTmp.y * scale);
    tmp.z = cuda_cast<int8_t>(floatTmp.z * scale);
    tmp.w = cuda_cast<int8_t>(floatTmp.w * scale);
    dst[idx] = tmp;
  }
}

__global__ static void quantizedKernel(char4* dst, half2 const* src, int64_t const sizeDiv4, float const* scalePtr) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x) {
    float const scale = __ldg(scalePtr);
    char4 tmp;
    int srcId = idx << 1;

    uint2 const h2 = __ldg(reinterpret_cast<uint2 const*>(src + srcId));

    half2 const half2Tmp = reinterpret_cast<half2 const&>(h2.x);
    half2 const half2Tmp2 = reinterpret_cast<half2 const&>(h2.y);

    tmp.x = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.x) * scale);
    tmp.y = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.y) * scale);
    tmp.z = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.x) * scale);
    tmp.w = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.y) * scale);
    dst[idx] = tmp;
  }
}

#ifdef ENABLE_BF16
__global__ static void quantizedKernel(
    char4* dst, __nv_bfloat162 const* src, int64_t const sizeDiv4, float const* scalePtr) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x) {
    float const scale = __ldg(scalePtr);
    char4 tmp;
    int srcId = idx << 1;

    uint2 const h2 = __ldg(reinterpret_cast<uint2 const*>(src + srcId));

    __nv_bfloat162 const bfloat162Tmp = reinterpret_cast<__nv_bfloat162 const&>(h2.x);
    __nv_bfloat162 const bfloat162Tmp2 = reinterpret_cast<__nv_bfloat162 const&>(h2.y);

    tmp.x = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp.x) * scale);
    tmp.y = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp.y) * scale);
    tmp.z = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp2.x) * scale);
    tmp.w = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp2.y) * scale);

    dst[idx] = tmp;
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int NUM_ELTS>
struct DstVec {
  static_assert("not implemented.");
};

template <>
struct DstVec<float2, 2> {
  using Type = uint32_t;
};

template <>
struct DstVec<half2, 4> {
  using Type = uint2;
};

#ifdef ENABLE_BF16

template <>
struct DstVec<__nv_bfloat162, 4> {
  using Type = uint2;
};

#endif  // ENABLE_BF16

template <typename T>
struct DstVec<T, 4> {
  static_assert(sizeof(T) == 4, "not implemented.");
  using Type = uint32_t;
};

template <typename T>
struct DstVec<T, 8> {
  static_assert(sizeof(T) == 2, "not implemented.");
  using Type = uint2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function of getting the absMax of all elements in the vector after clamping.
// Pack two elements in order to use possible hmax2 instructions.
template <typename T>
inline __device__ void clampAndAbsMax(T& localMax, uint4& vec, T const clampMin, T const clampMax) {
  static constexpr int NUM_ELTS = sizeof(uint4) / sizeof(T);

#pragma unroll
  for (int i = 0; i < NUM_ELTS; ++i) {
    T& val = reinterpret_cast<T*>(&vec)[i];
    val = cuda_clamp(val, clampMin, clampMax);
    localMax = cuda_max(localMax, cuda_abs(val));
  }
}

// Helper function of quantizing the vector and storing it to global memory.
// Pack two elements in order to use fast convert instructions.
template <typename T, typename QuantT, bool USE_SMEM>
inline __device__ void quantizeAndStore(
    QuantT* dstPtr, uint4 vec, T const clampMin, T const clampMax, float const scaleOrigQuant) {
  static constexpr int NUM_ELTS = sizeof(uint4) / sizeof(T);

  using DstVecType = typename DstVec<T, NUM_ELTS>::Type;
  DstVecType dstVec;
#pragma unroll
  for (int i = 0; i < NUM_ELTS; ++i) {
    T val = reinterpret_cast<T*>(&vec)[i];
    // Values loaded from smem has already been clamped.
    if constexpr (!USE_SMEM) {
      val = cuda_clamp(val, clampMin, clampMax);
    }
    float2 val2 = cuda_cast<float2>(val);
    val2.x *= scaleOrigQuant;
    val2.y *= scaleOrigQuant;
    QuantT quantVal = cuda_cast<QuantT>(val2);
    reinterpret_cast<QuantT*>(&dstVec)[i] = quantVal;
  }
  // Store to destination buffer.
  *reinterpret_cast<DstVecType*>(dstPtr) = dstVec;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename QuantT, bool USE_SMEM>
__global__ void perTokenQuantization(QuantT* dst, T const* src, int64_t const numRows, int64_t const numCols,
                                     float const* clampPtr, float* scalePtr, float* sumPtr, bool hasFp8MinScaling) {
  // Smem buffer.
  extern __shared__ uint4 smemBuffer[];

  // The clamping minimum / maximum values.
  T const clampMin = cuda_cast<T>(clampPtr ? clampPtr[0] : -FLT_MAX);
  T const clampMax = cuda_cast<T>(clampPtr ? clampPtr[1] : FLT_MAX);

  // Pack two elements in order to use higher through instructions.
  using T2 = typename packed_as<T, 2>::type;
  using QuantT2 = typename packed_as<QuantT, 2>::type;
  T2 const clampMin2 = cuda_cast<T2, T>(clampMin);
  T2 const clampMax2 = cuda_cast<T2, T>(clampMax);

  // The quantized data type's maximum value (upper-bound).
  static constexpr float MAX_QUANT_VAL = QuantTypeStaticVals<QuantT>::MAX_VAL;
  // The minimum scaling factor (lower-bound).
  static constexpr float MIN_SCALING_FACTOR = QuantTypeStaticVals<QuantT>::MIN_SCALING_FACTOR;
  static constexpr float MIN_SCALING_FACTOR_RCP = QuantTypeStaticVals<QuantT>::MIN_SCALING_FACTOR_RCP;

  // The number of elements in the packed uint4 vec.
  static constexpr int NUM_ELTS_PER_VEC = sizeof(uint4) / sizeof(T);
  static constexpr int NUM_ELTS2_PER_VEC = sizeof(uint4) / sizeof(T2);

  // The number of vectors in the column.
  int const numColVecs = numCols / NUM_ELTS_PER_VEC;
  // The vector pointers for src.
  uint4 const* srcVec = reinterpret_cast<uint4 const*>(src) + blockIdx.x * numColVecs;
  // The pointer for dst.
  QuantT* dstRow = dst + blockIdx.x * numCols;
  // T const* srcRow = src + blockIdx.x * numCols;

  T2 localMax2 = cuda_cast<T2, T>(T(1e-6f));
  float2 localSum2 = {0.f, 0.f};

  for (int i = threadIdx.x; i < numColVecs; i += blockDim.x) {
    uint4 vec = srcVec[i];

#pragma unroll
    for (int j = 0; j < NUM_ELTS2_PER_VEC; ++j) {
      T2& val2 = reinterpret_cast<T2*>(&vec)[j];
      val2 = cuda_clamp(val2, clampMin2, clampMax2);
      localMax2 = cuda_max(localMax2, cuda_abs(val2));
      // TODO: template the version that requires sum to avoid dynamic branching.
      if (sumPtr != nullptr) {
        localSum2.x += cuda_cast<float>(val2.x);
        localSum2.y += cuda_cast<float>(val2.y);
      }
    }
    // Avoid reloading from global memory.
    if constexpr (USE_SMEM) {
      smemBuffer[i] = vec;
    }
  }
  float const rowMax = blockAllReduceMax(cuda_cast<float>(cuda_max<T, T2>(localMax2)));
  if (threadIdx.x == 0) {
    scalePtr[blockIdx.x] = hasFp8MinScaling ? cuda_max(rowMax / MAX_QUANT_VAL, MIN_SCALING_FACTOR) : (rowMax / MAX_QUANT_VAL);
  }

  if (sumPtr != nullptr) {
    float rowSum[1] = {cuda_sum<float>(localSum2)};
    blockReduceSumV2<float, 1>(rowSum);
    if (threadIdx.x == 0) {
      sumPtr[blockIdx.x] = rowSum[0];
    }
  }

  float const scaleOrigQuant = hasFp8MinScaling ? fminf(MAX_QUANT_VAL / rowMax, MIN_SCALING_FACTOR_RCP) : MAX_QUANT_VAL / rowMax;
  for (int i = threadIdx.x; i < numColVecs; i += blockDim.x) {
    uint4 vec = USE_SMEM ? smemBuffer[i] : srcVec[i];
    QuantT2* dstPtr = reinterpret_cast<QuantT2*>(dstRow + i * NUM_ELTS_PER_VEC);
    quantizeAndStore<T2, QuantT2, USE_SMEM>(dstPtr, vec, clampMin2, clampMax2, scaleOrigQuant);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP4 Quantization

constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;
constexpr int CVT_FP4_THREADS_PER_WARP = 32;
constexpr int CVT_FP8_TO_FP4_ELTS_PER_THREAD = 16;

// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]), "f"(array[4]), "f"(array[5]), "f"(array[6]),
        "f"(array[7]));
  return val;
#else
  // static_assert(false, "not supported.");
  return 0;
#endif
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y), "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  // static_assert(false, "not supported.");
  return 0;
#endif
}

// Convert 8 float2 values into 16 e2m1 values (represented as one uint64_t).
inline __device__ uint64_t fp32_vec_to_e2m1(float2 (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint64_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      ".reg .b8 byte4;\n"
      ".reg .b8 byte5;\n"
      ".reg .b8 byte6;\n"
      ".reg .b8 byte7;\n"
      ".reg .b32 val0;\n"
      ".reg .b32 val1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0,  %2,  %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1,  %4,  %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2,  %6,  %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3,  %8,  %7;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte4, %10,  %9;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte5, %12, %11;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte6, %14, %13;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte7, %16, %15;\n"
      "mov.b32 val0, {byte0, byte1, byte2, byte3};\n"
      "mov.b32 val1, {byte4, byte5, byte6, byte7};\n"
      "mov.b64 %0, {val0, val1};\n"
      "}"
      : "=l"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y), "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y), "f"(array[4].x), "f"(array[4].y), "f"(array[5].x), "f"(array[5].y),
        "f"(array[6].x), "f"(array[6].y), "f"(array[7].x), "f"(array[7].y));
  return val;
#else
  // static_assert(false, "not supported.");
  return 0;
#endif
}

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Define a 16 bytes packed data type.
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];
  static_assert(sizeof(elts) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vector size should match the number of elements per thread.");
};

template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];
  static_assert(sizeof(elts) == sizeof(__nv_fp8_e4m3) * CVT_FP8_TO_FP4_ELTS_PER_THREAD,
                "Vector size should match the number of elements per thread.");
};

// Convert 4 float2 values into 8 e4m3 values (represented as one uint64_t).
inline __device__ uint64_t fp32_vec_to_e4m3(float2 (&array)[4]) {
  union {
    uint64_t val;
    __nv_fp8x2_e4m3 elts[4];
  } u;

  static_assert(sizeof(u.val) == sizeof(u.elts), "Expected to alias uint64_t and __nv_fp8x2_e4m3[4]");

  u.elts[0] = __nv_fp8x2_e4m3(array[0]);
  u.elts[1] = __nv_fp8x2_e4m3(array[1]);
  u.elts[2] = __nv_fp8x2_e4m3(array[2]);
  u.elts[3] = __nv_fp8x2_e4m3(array[3]);
  return u.val;
}

// Quantizes the provided PackedVec into the uint64_t output
template <class Type, int SF_VEC_SIZE>
__device__ uint64_t cvt_warp_fp16_to_mxfp8(PackedVec<Type>& vec, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Get absolute maximum values among the local 8 values.
  auto localMax = cuda_abs(vec.elts[0]);

// Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = cuda_max(localMax, cuda_abs(vec.elts[i]));
  }

  constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
  // Get the absolute maximum among all 16 values (two threads for 16, four threads for 32).
  localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  if constexpr (CVT_NUM_THREADS_PER_SF == 4) {
    localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
  }
  // Get the final absolute maximum values.
  float vecMax = float(cuda_max(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of mxfp8).
  float SFValue = vecMax * reciprocal_approximate_ftz(448.0f);
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  __nv_fp8_e8m0 tmpSFVal;
  tmpSFVal.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
  float SFValueNarrow = static_cast<float>(tmpSFVal);
  fp8SFVal = tmpSFVal.__x;
  // Get the output scale (reciprocal of the SFValue).
  float outputScale = SFValue != 0.f ? reciprocal_approximate_ftz(SFValueNarrow) : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e4m3 values.
  uint64_t e4m3Vec = fp32_vec_to_e4m3(fp2Vals);

  // Write the e4m3 values to global memory.
  return e4m3Vec;
#else
  return 0;
#endif
}

// Quantizes the provided PackedVec into the uint32_t output
template <class Type, int SF_VEC_SIZE, bool UE8M0_SF>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Get absolute maximum values among the local 8 values.
  auto localMax = cuda_abs(vec.elts[0]);

// Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = cuda_max(localMax, cuda_abs(vec.elts[i]));
  }

  constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
  // Get the absolute maximum among all 16 values (two threads for 16, four threads for 32).
  localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  if constexpr (CVT_NUM_THREADS_PER_SF == 4) {
    localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
  }
  // Get the final absolute maximum values.
  float vecMax = float(cuda_max(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of e2m1).
  // maximum value of e2m1 = 6.0.
  // TODO: use half as compute data type.
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  float SFValueNarrow;
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  if constexpr (UE8M0_SF) {
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValueNarrow = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
  } else {
    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValueNarrow = static_cast<float>(tmp);
  }
  // Get the output scale.
  // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) * reciprocal(SFScaleVal))
  float outputScale = SFValue != 0 ? reciprocal_approximate_ftz(SFValueNarrow * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
#else
  return 0;
#endif
}

template <class Type, int SF_VEC_SIZE, bool UE8M0_SF>
__device__ uint64_t cvt_warp_fp8_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  float const dequant_to_fp16_scale = 6.f * reciprocal_approximate_ftz(SFScaleVal);

  // Dequant fp8 to fp16
  __half2 vec_half2[8];
#pragma unroll
  for (int i = 0; i < CVT_FP8_TO_FP4_ELTS_PER_THREAD / 2; i++) {
    float2 tmp = static_cast<float2>(vec.elts[i]);
    tmp.x *= dequant_to_fp16_scale;
    tmp.y *= dequant_to_fp16_scale;
    vec_half2[i] = __float22half2_rn(tmp);
  }

  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec_half2[0]);
  // Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_FP8_TO_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec_half2[i]));
  }

  constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_FP8_TO_FP4_ELTS_PER_THREAD;
  if constexpr (CVT_NUM_THREADS_PER_SF == 2) {
    // For block 32, we need to reduce the local max across two threads.
    localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  }

  // Get the final absolute maximum values.
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of e2m1).
  // maximum value of e2m1 = 6.0.
  // TODO: use half as compute data type.
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  if constexpr (UE8M0_SF) {
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
  } else {
    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
  }
  // Get the output scale.
  // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) * reciprocal(SFScaleVal))
  float outputScale = SFValue != 0 ? SFScaleVal * reciprocal_approximate_ftz(SFValue) : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_FP8_TO_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP8_TO_FP4_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i] = __half22float2(vec_half2[i]);
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  uint64_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
#else
  return 0;
#endif
}

template <int SF_VEC_SIZE>
inline __device__ __host__ int64_t get_sf_out_offset_128x4(
    std::optional<int> batchIdx, int mIdx, int kIdx, std::optional<int> numRows, int numCols) {
  // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

  // batched tensor
  // SF layout [numBTiles, numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  // --> index [bTileIdx, mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

  int32_t innerKIdx = (kIdx % 4);
  int64_t innerKStride = 1;

  int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
  int64_t innerMStride = 4 * innerKStride;  // 4

  // M tile layout [32, 4] is column-major.
  int32_t outerMIdx = (mIdx % 32);
  int64_t outerMStride = 4 * innerMStride;  // 16

  int32_t kTileIdx = (kIdx / 4);
  int64_t kTileStride = 32 * outerMStride;  // 512

  // SF vector size 16. We round the "numCols" up to a multiple of 64.
  int factor = SF_VEC_SIZE * 4;
  int32_t numKTiles = (numCols + factor - 1) / factor;
  int32_t mTileIdx = mIdx / (32 * 4);
  int64_t mTileStride = numKTiles * kTileStride;

  // Each SF block has 128 rows so pad rows to the multiple of 128.
  int32_t numMTiles = (numRows.value_or(0) + 128 - 1) / 128;
  int64_t bTileStride = numMTiles * mTileStride;

  // Compute the global offset.
  int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride + kTileIdx * kTileStride + outerMIdx * outerMStride + innerMIdx * innerMStride + innerKIdx * innerKStride;

  return SFOffset;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF, int SF_VEC_SIZE>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(std::optional<int> batchIdx, int rowIdx, int colIdx,
                                                       std::optional<int> numRows, int numCols, SFType* SFout, FP4QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(
      CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2 || CVT_FP4_NUM_THREADS_PER_SF == 4);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    if (layout == FP4QuantizationSFLayout::SWIZZLED) {
      // SF vector index (16 elements share one SF in the K dimension).
      // numRows and numCols are unpadded.
      int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
      int32_t mIdx = rowIdx;

      auto SFOffset = get_sf_out_offset_128x4<SF_VEC_SIZE>(batchIdx, mIdx, kIdx, numRows, numCols);
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else if (layout == FP4QuantizationSFLayout::LINEAR) {
      // Linear row-major layout, no padding required.
      int32_t KTileIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;

      int32_t numKTiles = numCols / SF_VEC_SIZE;
      int64_t mTileStride = numKTiles;

      int64_t BTileStride = numRows.value_or(0) * mTileStride;

      int64_t SFOffset = batchIdx.value_or(0) * BTileStride + rowIdx * mTileStride + KTileIdx;
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else {
      return nullptr;
    }
  }
#endif
  return nullptr;
}

// Use UE4M3 by default.
template <class Type, int SF_VEC_SIZE, bool UE8M0_SF>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4_3d(
#else
cvt_fp16_to_fp4_3d(
#endif
    int32_t numbatches, int32_t numRows, int32_t numCols, Type const* in, float const* SFScale, uint32_t* out,
    uint32_t* SFout, FP4QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;  // 2 or 4
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is (448.f / (Alpha_A / 6.f)).
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  asm volatile("griddepcontrol.wait;");
  // Input tensor batch/row/col loops.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int batchIdx = 0; batchIdx < numbatches; batchIdx++) {
      for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD; colIdx += blockDim.x) {
        int64_t inOffset = batchIdx * numRows * (numCols / CVT_FP4_ELTS_PER_THREAD) + rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
        PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
        // Get the output tensor offset.
        // Same as inOffset because 8 elements are packed into one uint32_t.
        int64_t outOffset = inOffset;
        auto& out_pos = out[outOffset];

        std::optional<int> optionalBatchIdx = batchIdx;
        std::optional<int> optionalNumRows = numRows;

        auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF, SF_VEC_SIZE>(
            optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numCols, SFout, layout);

        out_pos = cvt_warp_fp16_to_fp4<Type, SF_VEC_SIZE, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
      }
    }
  }
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Use UE4M3 by default.
template <int SF_VEC_SIZE, bool UE8M0_SF>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp8_to_fp4_3d(
#else
cvt_fp8_to_fp4_3d(
#endif
    int32_t numbatches, int32_t numRows, int32_t numCols, __nv_fp8_e4m3 const* in, float const* SFScale,
    uint32_t* out, uint32_t* SFout, FP4QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<__nv_fp8_e4m3>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_FP8_TO_FP4_ELTS_PER_THREAD;
  static_assert(
      sizeof(PackedVec) == sizeof(__nv_fp8_e4m3) * CVT_FP8_TO_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is (448.f / (Alpha_A / 6.f)).
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  // Input tensor batch/row/col loops.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int batchIdx = 0; batchIdx < numbatches; batchIdx++) {
      for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP8_TO_FP4_ELTS_PER_THREAD; colIdx += blockDim.x) {
        int64_t inOffset = batchIdx * numRows * (numCols / CVT_FP4_ELTS_PER_THREAD) + rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
        PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
        // Get the output tensor offset.
        // Same as inOffset because 16 elements are packed into one uint64_t.
        int64_t outOffset = inOffset;
        auto& out_pos = out[outOffset];

        std::optional<int> optionalBatchIdx = batchIdx;
        std::optional<int> optionalNumRows = numRows;

        auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF, SF_VEC_SIZE>(
            optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numCols, SFout, layout);

        out_pos = cvt_warp_fp8_to_fp4<__nv_fp8_e4m3, SF_VEC_SIZE, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
      }
    }
  }
#endif
}

// Use UE4M3 by default.
template <class Type, int SF_VEC_SIZE, bool UE8M0_SF>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4(
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows, int32_t numCols, Type const* in, float const* SFScale, uint32_t* out, uint32_t* SFout,
    FP4QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is (448.f / (Alpha_A / 6.f)).
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  asm volatile("griddepcontrol.wait;");
  // Input tensor row/col loops.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD; colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      // Get the output tensor offset.
      // Same as inOffset because 8 elements are packed into one uint32_t.
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF, SF_VEC_SIZE>(
          std::nullopt /* batchIdx */, rowIdx, colIdx, std::nullopt /* numRows */, numCols, SFout, layout);

      out_pos = cvt_warp_fp16_to_fp4<Type, SF_VEC_SIZE, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Use UE4M3 by default.
template <int SF_VEC_SIZE, bool UE8M0_SF>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp8_to_fp4(
#else
cvt_fp8_to_fp4(
#endif
    int32_t numRows, int32_t numCols, __nv_fp8_e4m3 const* in, float const* SFScale, uint64_t* out, uint32_t* SFout,
    FP4QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<__nv_fp8_e4m3>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_FP8_TO_FP4_ELTS_PER_THREAD;
  static_assert(
      sizeof(PackedVec) == sizeof(__nv_fp8_e4m3) * CVT_FP8_TO_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is (448.f / (Alpha_A / 6.f)).
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  // Input tensor row/col loops.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP8_TO_FP4_ELTS_PER_THREAD; colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP8_TO_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      // Get the output tensor offset.
      // Same as inOffset because 16 elements are packed into one uint64_t.
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF, SF_VEC_SIZE>(
          std::nullopt /* batchIdx */, rowIdx, colIdx, std::nullopt /* numRows */, numCols, SFout, layout);

      out_pos = cvt_warp_fp8_to_fp4<__nv_fp8_e4m3, SF_VEC_SIZE, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
#endif
}

__global__ void nvfp4_block_scale_interleave_kernel(
    int numbatches, int numRows, int numCols, uint8_t const* SFIn, uint8_t* SFOutput);
}  // namespace kernels
}  // namespace onnxruntime::llm
