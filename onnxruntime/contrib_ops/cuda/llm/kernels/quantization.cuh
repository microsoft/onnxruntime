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
#pragma once

#include "contrib_ops/cuda/llm/common/cuda_type_utils.cuh"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"

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

}  // namespace kernels
}  // namespace onnxruntime::llm
