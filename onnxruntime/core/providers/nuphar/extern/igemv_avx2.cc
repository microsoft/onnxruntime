// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "igemv_avx2.h"

#ifdef _WIN32
// A temp fix of visual studio bug
#pragma warning(push)
#pragma warning(disable : 4310)
#endif  // _WIN32

namespace onnxruntime {

#ifdef NUPHAR_USE_AVX2

// General macros for avx2 integer gemv
#define mm256_load_vec_epi8(Reg, Idx) \
  auto Reg##Idx = _mm256_lddqu_si256((const __m256i*)(vec + (Idx)*32));

#define mm256_load_vec_epi16(Reg, Idx) \
  auto Reg##Idx = _mm256_lddqu_si256((const __m256i*)(vec + (Idx)*16));

#define mm256_accumulate_epi32(Reg)      \
  Reg = _mm256_hadd_epi32(Reg, Reg);     \
  Reg = _mm256_hadd_epi32(Reg, Reg);     \
  auto res = _mm_add_epi32(              \
      _mm256_castsi256_si128(Reg),       \
      _mm256_extractf128_si256(Reg, 1)); \
  *(int*)(output + i) = _mm_cvtsi128_si32(res);

// Extended macros for avx2 integer gemv to handle non-padded dimension
#define mm256_set_mask_epi8(rest, h, l)                                                                                       \
  switch (rest) {                                                                                                             \
    case 0:                                                                                                                   \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h); \
      break;                                                                                                                  \
    case 1:                                                                                                                   \
      mask = _mm256_set_epi8(h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 2:                                                                                                                   \
      mask = _mm256_set_epi8(h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 3:                                                                                                                   \
      mask = _mm256_set_epi8(h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 4:                                                                                                                   \
      mask = _mm256_set_epi8(h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 5:                                                                                                                   \
      mask = _mm256_set_epi8(h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 6:                                                                                                                   \
      mask = _mm256_set_epi8(h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 7:                                                                                                                   \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 8:                                                                                                                   \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 9:                                                                                                                   \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 10:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 11:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 12:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 13:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 14:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 15:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 16:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 17:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 18:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 19:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 20:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 21:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 22:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 23:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 24:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 25:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 26:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l, l); \
      break;                                                                                                                  \
    case 27:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l, l); \
      break;                                                                                                                  \
    case 28:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l, l); \
      break;                                                                                                                  \
    case 29:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l, l); \
      break;                                                                                                                  \
    case 30:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l, l); \
      break;                                                                                                                  \
    case 31:                                                                                                                  \
      mask = _mm256_set_epi8(h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, l); \
      break;                                                                                                                  \
    case 32:                                                                                                                  \
      mask = _mm256_set_epi8(l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l); \
      break;                                                                                                                  \
    default:                                                                                                                  \
      printf("Mask set is wrong! Please double check!");                                                                      \
      abort();                                                                                                                \
  }

#define mm256_mask_load_vec_epi8(Reg, Idx, Offset, Mask) \
  auto Reg##Idx = _mm256_blendv_epi8(                    \
      zero, _mm256_lddqu_si256((const __m256i*)(vec + (Idx)*32 - (Offset))), Mask);

#define mm256_mat_vec_off_mul_epi8(Reg, Idx, Offset)                                   \
  tmp = _mm256_maddubs_epi16(                                                          \
      (Reg##Idx), _mm256_lddqu_si256((const __m256i*)(rowPtr + (Idx)*32 - (Offset)))); \
  sum = _mm256_add_epi32(sum, _mm256_madd_epi16(tmp, one));

#define mm256_mat_vec_mul_epi8(Reg, Idx) mm256_mat_vec_off_mul_epi8(Reg, Idx, 0)

#define mm256_set_mask_epi16(Reg, Idx, Remain, src)                 \
  switch (Remain) {                                                 \
    case 1: {                                                       \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0x80);          \
    } break;                                                        \
    case 2: {                                                       \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xc0);          \
    } break;                                                        \
    case 3: {                                                       \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xe0);          \
    } break;                                                        \
    case 4: {                                                       \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xf0);          \
    } break;                                                        \
    case 5: {                                                       \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xf8);          \
    } break;                                                        \
    case 6: {                                                       \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xfc);          \
    } break;                                                        \
    case 7: {                                                       \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xfe);          \
    } break;                                                        \
    case 8: {                                                       \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xff);          \
    } break;                                                        \
    case 9: {                                                       \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0x80);          \
      Reg##Idx = _mm256_insertf128_si256(Reg##Idx, Reg##Idx##h, 1); \
    } break;                                                        \
    case 10: {                                                      \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xc0);          \
      Reg##Idx = _mm256_insertf128_si256(Reg##Idx, Reg##Idx##h, 1); \
    } break;                                                        \
    case 11: {                                                      \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xe0);          \
      Reg##Idx = _mm256_insertf128_si256(Reg##Idx, Reg##Idx##h, 1); \
    } break;                                                        \
    case 12: {                                                      \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xf0);          \
      Reg##Idx = _mm256_insertf128_si256(Reg##Idx, Reg##Idx##h, 1); \
    } break;                                                        \
    case 13: {                                                      \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xf8);          \
      Reg##Idx = _mm256_insertf128_si256(Reg##Idx, Reg##Idx##h, 1); \
    } break;                                                        \
    case 14: {                                                      \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xfc);          \
      Reg##Idx = _mm256_insertf128_si256(Reg##Idx, Reg##Idx##h, 1); \
    } break;                                                        \
    case 15: {                                                      \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xfe);          \
      Reg##Idx = _mm256_insertf128_si256(Reg##Idx, Reg##Idx##h, 1); \
    } break;                                                        \
    case 16: {                                                      \
      Reg##Idx = _mm256_blend_epi16(zero, src##Idx, 0xff);          \
      Reg##Idx = _mm256_insertf128_si256(Reg##Idx, Reg##Idx##h, 1); \
    } break;                                                        \
    default:                                                        \
      abort();                                                      \
  }

#define mm256_mask_load_vec_epi16(Reg, Idx, Offset, Remain)                        \
  auto Reg##Idx = _mm256_lddqu_si256((const __m256i*)(vec + (Idx)*16 - (Offset))); \
  if (Remain != 0) {                                                               \
    auto Reg##Idx##l = _mm256_castsi256_si128(Reg##Idx);                           \
    auto Reg##Idx##h = _mm256_extractf128_si256(Reg##Idx, 1);                      \
    __m256i m256_##Idx;                                                            \
    if (Remain <= 8) {                                                             \
      m256_##Idx = _mm256_castsi128_si256(hzero);                                  \
      m256_##Idx = _mm256_insertf128_si256(m256_##Idx, Reg##Idx##h, 1);            \
    } else {                                                                       \
      m256_##Idx = _mm256_castsi128_si256(Reg##Idx##l);                            \
      m256_##Idx = _mm256_insertf128_si256(m256_##Idx, hzero, 1);                  \
    }                                                                              \
    mm256_set_mask_epi16(Reg, Idx, Remain, m256_);                                 \
  }

#define mm256_mat_vec_off_mul_epi16(Reg, Idx, Offset)                                  \
  tmp = _mm256_madd_epi16(                                                             \
      (Reg##Idx), _mm256_lddqu_si256((const __m256i*)(rowPtr + (Idx)*16 - (Offset)))); \
  sum = _mm256_add_epi32(sum, tmp);

#define mm256_mat_vec_mul_epi16(Reg, Idx) mm256_mat_vec_off_mul_epi16(Reg, Idx, 0)

// S8U8S32: int8_t * uint8_t = int32_t; S16S16S32: int16_t * int16_t = int32_t
// R stands for row major
void AVX2IntGemvS8U8S32R(
    int8_t* matrix,
    uint8_t* vec,
    int matrixRowDimension,
    int paddedRowDimension,
    int matrixColumnDimension,
    int32_t* output) {
  __m256i one = _mm256_set1_epi16(1);
  __m256i tmp = _mm256_setzero_si256();

  int rowUnrollFactor = (matrixRowDimension + 31) / 32;
  switch (rowUnrollFactor) {
    case 1: {
      mm256_load_vec_epi8(r, 0);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * paddedRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 2: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * paddedRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 3: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      mm256_load_vec_epi8(r, 2);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * paddedRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_mat_vec_mul_epi8(r, 2);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 4: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      mm256_load_vec_epi8(r, 2);
      mm256_load_vec_epi8(r, 3);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * paddedRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_mat_vec_mul_epi8(r, 2);
        mm256_mat_vec_mul_epi8(r, 3);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 5: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      mm256_load_vec_epi8(r, 2);
      mm256_load_vec_epi8(r, 3);
      mm256_load_vec_epi8(r, 4);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * paddedRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_mat_vec_mul_epi8(r, 2);
        mm256_mat_vec_mul_epi8(r, 3);
        mm256_mat_vec_mul_epi8(r, 4);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 6: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      mm256_load_vec_epi8(r, 2);
      mm256_load_vec_epi8(r, 3);
      mm256_load_vec_epi8(r, 4);
      mm256_load_vec_epi8(r, 5);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * paddedRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_mat_vec_mul_epi8(r, 2);
        mm256_mat_vec_mul_epi8(r, 3);
        mm256_mat_vec_mul_epi8(r, 4);
        mm256_mat_vec_mul_epi8(r, 5);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 7: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      mm256_load_vec_epi8(r, 2);
      mm256_load_vec_epi8(r, 3);
      mm256_load_vec_epi8(r, 4);
      mm256_load_vec_epi8(r, 5);
      mm256_load_vec_epi8(r, 6);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * paddedRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_mat_vec_mul_epi8(r, 2);
        mm256_mat_vec_mul_epi8(r, 3);
        mm256_mat_vec_mul_epi8(r, 4);
        mm256_mat_vec_mul_epi8(r, 5);
        mm256_mat_vec_mul_epi8(r, 6);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 8: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      mm256_load_vec_epi8(r, 2);
      mm256_load_vec_epi8(r, 3);
      mm256_load_vec_epi8(r, 4);
      mm256_load_vec_epi8(r, 5);
      mm256_load_vec_epi8(r, 6);
      mm256_load_vec_epi8(r, 7);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * paddedRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_mat_vec_mul_epi8(r, 2);
        mm256_mat_vec_mul_epi8(r, 3);
        mm256_mat_vec_mul_epi8(r, 4);
        mm256_mat_vec_mul_epi8(r, 5);
        mm256_mat_vec_mul_epi8(r, 6);
        mm256_mat_vec_mul_epi8(r, 7);
        mm256_accumulate_epi32(sum);
      }
    } break;
    default: {
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * paddedRowDimension;
        uint8_t* vecPtr = vec;

        __m256i sum = _mm256_setzero_si256();
        int j = 0;
        for (; j < std::min(paddedRowDimension, paddedRowDimension / 64 * 64); j += 64) {
          tmp = _mm256_maddubs_epi16(
              _mm256_lddqu_si256((const __m256i*)(vecPtr + j)),
              _mm256_lddqu_si256((const __m256i*)(rowPtr + j)));
          sum = _mm256_add_epi32(sum, _mm256_madd_epi16(tmp, one));

          tmp = _mm256_maddubs_epi16(
              _mm256_lddqu_si256((const __m256i*)(vecPtr + 32 + j)),
              _mm256_lddqu_si256((const __m256i*)(rowPtr + 32 + j)));
          sum = _mm256_add_epi32(sum, _mm256_madd_epi16(tmp, one));
        }
        if (paddedRowDimension % 64) {
          tmp = _mm256_maddubs_epi16(
              _mm256_lddqu_si256((const __m256i*)(vecPtr + j)),
              _mm256_lddqu_si256((const __m256i*)(rowPtr + j)));
          sum = _mm256_add_epi32(sum, _mm256_madd_epi16(tmp, one));
        }

        mm256_accumulate_epi32(sum);
      }
    }
  }
}

void AVX2IntGemvS16S16S32R(
    int16_t* matrix,
    int16_t* vec,
    int matrixRowDimension,
    int paddedRowDimension,
    int matrixColumnDimension,
    int32_t* output) {
  __m256i tmp = _mm256_setzero_si256();

  int rowUnrollFactor = (matrixRowDimension + 15) / 16;
  switch (rowUnrollFactor) {
    case 1: {
      mm256_load_vec_epi16(r, 0);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int16_t* rowPtr = matrix + i * paddedRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi16(r, 0);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 2: {
      mm256_load_vec_epi16(r, 0);
      mm256_load_vec_epi16(r, 1);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int16_t* rowPtr = matrix + i * paddedRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi16(r, 0);
        mm256_mat_vec_mul_epi16(r, 1);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 3: {
      mm256_load_vec_epi16(r, 0);
      mm256_load_vec_epi16(r, 1);
      mm256_load_vec_epi16(r, 2);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int16_t* rowPtr = matrix + i * paddedRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi16(r, 0);
        mm256_mat_vec_mul_epi16(r, 1);
        mm256_mat_vec_mul_epi16(r, 2);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 4: {
      mm256_load_vec_epi16(r, 0);
      mm256_load_vec_epi16(r, 1);
      mm256_load_vec_epi16(r, 2);
      mm256_load_vec_epi16(r, 3);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int16_t* rowPtr = matrix + i * paddedRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi16(r, 0);
        mm256_mat_vec_mul_epi16(r, 1);
        mm256_mat_vec_mul_epi16(r, 2);
        mm256_mat_vec_mul_epi16(r, 3);
        mm256_accumulate_epi32(sum);
      }
    } break;
    default: {
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int16_t* rowPtr = matrix + i * paddedRowDimension;
        int16_t* vecPtr = vec;

        __m256i sum = _mm256_setzero_si256();
        for (int j = 0; j < paddedRowDimension; j += 16) {
          tmp = _mm256_madd_epi16(
              _mm256_lddqu_si256((const __m256i*)(vecPtr + j)),
              _mm256_lddqu_si256((const __m256i*)(rowPtr + j)));
          sum = _mm256_add_epi32(sum, tmp);
        }

        mm256_accumulate_epi32(sum);
      }
    }
  }
}

// Ex stands for extended to handle non-padded dimension
void AVX2IntGemvS8U8S32REx(
    int8_t* matrix,
    uint8_t* vec,
    int matrixRowDimension,
    int matrixColumnDimension,
    int32_t* output) {
  __m256i zero = _mm256_setzero_si256();
  __m256i one = _mm256_set1_epi16(1);
  __m256i tmp = _mm256_setzero_si256();
  int rowUnrollFactor = (matrixRowDimension + 31) / 32;
  int matrixRowRemain = matrixRowDimension % 32;
  int matrixRowOffset = (32 - matrixRowRemain) % 32;

  __m256i mask;
  if (rowUnrollFactor > 1) {
    mm256_set_mask_epi8(matrixRowRemain, -128, 0);
  } else if (matrixRowOffset == 0) {
    mm256_set_mask_epi8(32, 0, -128);
  } else {
    mm256_set_mask_epi8(matrixRowOffset, 0, -128);
  }

  switch (rowUnrollFactor) {
    case 1: {
      mm256_mask_load_vec_epi8(r, 0, 0, mask);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_off_mul_epi8(r, 0, 0);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 2: {
      mm256_load_vec_epi8(r, 0);
      mm256_mask_load_vec_epi8(r, 1, matrixRowOffset, mask);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_off_mul_epi8(r, 1, matrixRowOffset);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 3: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      mm256_mask_load_vec_epi8(r, 2, matrixRowOffset, mask);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_mat_vec_off_mul_epi8(r, 2, matrixRowOffset);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 4: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      mm256_load_vec_epi8(r, 2);
      mm256_mask_load_vec_epi8(r, 3, matrixRowOffset, mask);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_mat_vec_mul_epi8(r, 2);
        mm256_mat_vec_off_mul_epi8(r, 3, matrixRowOffset);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 5: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      mm256_load_vec_epi8(r, 2);
      mm256_load_vec_epi8(r, 3);
      mm256_mask_load_vec_epi8(r, 4, matrixRowOffset, mask);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_mat_vec_mul_epi8(r, 2);
        mm256_mat_vec_mul_epi8(r, 3);
        mm256_mat_vec_off_mul_epi8(r, 4, matrixRowOffset);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 6: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      mm256_load_vec_epi8(r, 2);
      mm256_load_vec_epi8(r, 3);
      mm256_load_vec_epi8(r, 4);
      mm256_mask_load_vec_epi8(r, 5, matrixRowOffset, mask);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_mat_vec_mul_epi8(r, 2);
        mm256_mat_vec_mul_epi8(r, 3);
        mm256_mat_vec_mul_epi8(r, 4);
        mm256_mat_vec_off_mul_epi8(r, 5, matrixRowOffset);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 7: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      mm256_load_vec_epi8(r, 2);
      mm256_load_vec_epi8(r, 3);
      mm256_load_vec_epi8(r, 4);
      mm256_load_vec_epi8(r, 5);
      mm256_mask_load_vec_epi8(r, 6, matrixRowOffset, mask);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_mat_vec_mul_epi8(r, 2);
        mm256_mat_vec_mul_epi8(r, 3);
        mm256_mat_vec_mul_epi8(r, 4);
        mm256_mat_vec_mul_epi8(r, 5);
        mm256_mat_vec_off_mul_epi8(r, 6, matrixRowOffset);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 8: {
      mm256_load_vec_epi8(r, 0);
      mm256_load_vec_epi8(r, 1);
      mm256_load_vec_epi8(r, 2);
      mm256_load_vec_epi8(r, 3);
      mm256_load_vec_epi8(r, 4);
      mm256_load_vec_epi8(r, 5);
      mm256_load_vec_epi8(r, 6);
      mm256_mask_load_vec_epi8(r, 7, matrixRowOffset, mask);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi8(r, 0);
        mm256_mat_vec_mul_epi8(r, 1);
        mm256_mat_vec_mul_epi8(r, 2);
        mm256_mat_vec_mul_epi8(r, 3);
        mm256_mat_vec_mul_epi8(r, 4);
        mm256_mat_vec_mul_epi8(r, 5);
        mm256_mat_vec_mul_epi8(r, 6);
        mm256_mat_vec_off_mul_epi8(r, 7, matrixRowOffset);
        mm256_accumulate_epi32(sum);
      }
    } break;
    default: {
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int8_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        int j = 0, id = 0;
        for (; j < std::min(matrixRowDimension, matrixRowDimension / 32 * 32); j += 32, id++) {
          mm256_load_vec_epi8(reg, id);
          mm256_mat_vec_mul_epi8(reg, id);
        }
        if (matrixRowRemain) {
          mm256_mask_load_vec_epi8(reg, id, matrixRowOffset, mask);
          mm256_mat_vec_off_mul_epi8(reg, id, matrixRowOffset);
        }
        mm256_accumulate_epi32(sum);
      }
    }
  }
}

void AVX2IntGemvS16S16S32REx(
    int16_t* matrix,
    int16_t* vec,
    int matrixRowDimension,
    int matrixColumnDimension,
    int32_t* output) {
  __m128i hzero = _mm_setzero_si128();
  __m256i zero = _mm256_setzero_si256();
  __m256i tmp = _mm256_setzero_si256();
  int rowUnrollFactor = (matrixRowDimension + 15) / 16;
  int matrixRowRemain = matrixRowDimension % 16;
  int matrixRowOffset = (16 - matrixRowRemain) % 16;

  switch (rowUnrollFactor) {
    case 1: {
      mm256_mask_load_vec_epi16(r, 0, matrixRowOffset, matrixRowRemain);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int16_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_off_mul_epi16(r, 0, matrixRowOffset);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 2: {
      mm256_load_vec_epi16(r, 0);
      mm256_mask_load_vec_epi16(r, 1, matrixRowOffset, matrixRowRemain);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int16_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi16(r, 0);
        mm256_mat_vec_off_mul_epi16(r, 1, matrixRowOffset);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 3: {
      mm256_load_vec_epi16(r, 0);
      mm256_load_vec_epi16(r, 1);
      mm256_mask_load_vec_epi16(r, 2, matrixRowOffset, matrixRowRemain);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int16_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi16(r, 0);
        mm256_mat_vec_mul_epi16(r, 1);
        mm256_mat_vec_off_mul_epi16(r, 2, matrixRowOffset);
        mm256_accumulate_epi32(sum);
      }
    } break;
    case 4: {
      mm256_load_vec_epi16(r, 0);
      mm256_load_vec_epi16(r, 1);
      mm256_load_vec_epi16(r, 2);
      mm256_mask_load_vec_epi16(r, 3, matrixRowOffset, matrixRowRemain);
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int16_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        mm256_mat_vec_mul_epi16(r, 0);
        mm256_mat_vec_mul_epi16(r, 1);
        mm256_mat_vec_mul_epi16(r, 2);
        mm256_mat_vec_off_mul_epi16(r, 3, matrixRowOffset);
        mm256_accumulate_epi32(sum);
      }
    } break;
    default: {
      for (int i = 0; i < matrixColumnDimension; ++i) {
        int16_t* rowPtr = matrix + i * matrixRowDimension;
        __m256i sum = _mm256_setzero_si256();
        int j = 0, id = 0;
        for (; j < std::min(matrixRowDimension, matrixRowDimension / 16 * 16); j += 16, id++) {
          mm256_load_vec_epi16(reg, id);
          mm256_mat_vec_mul_epi16(reg, id);
        }
        mm256_mask_load_vec_epi16(reg, id, matrixRowOffset, matrixRowRemain);
        mm256_mat_vec_off_mul_epi16(reg, id, matrixRowOffset);
        mm256_accumulate_epi32(sum);
      }
    }
  }
}
#endif

}  // namespace onnxruntime

#ifdef _WIN32
#pragma warning(pop)
#endif
