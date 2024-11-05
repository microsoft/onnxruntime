/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qsoftmax_kernel_avx2.cpp

Abstract:

    This module implements the lookup-based quantized softmax kernels for x64 avx2.

--*/

#include "mlasi.h"

static uint8_t reduce_max_u8_avx2(const uint8_t* data, size_t size) {
  __m256i max_vec = _mm256_set1_epi8(0);

  size_t i;
  for (i = 0; i + 32 <= size; i += 32) {
    __m256i vec = _mm256_loadu_si256((const __m256i*)(data + i));
    max_vec = _mm256_max_epu8(max_vec, vec);
  }

  // Now reduce the 256-bit max_vec into a single max value
  // First, split the 256-bit vector into two 128-bit halves and compute the max
  // between them
  __m128i max_128 = _mm_max_epu8(_mm256_castsi256_si128(max_vec), _mm256_extracti128_si256(max_vec, 1));

  // Further reduce the 128-bit vector to a scalar
  // Extract the upper 64-bit part and compute the max
  max_128 = _mm_max_epu8(max_128, _mm_srli_si128(max_128, 8));
  // Extract the upper 32-bit part and compute the max
  max_128 = _mm_max_epu8(max_128, _mm_srli_si128(max_128, 4));
  // Extract the upper 16-bit part and compute the max
  max_128 = _mm_max_epu8(max_128, _mm_srli_si128(max_128, 2));
  // Extract the upper 8-bit part and compute the max
  max_128 = _mm_max_epu8(max_128, _mm_srli_si128(max_128, 1));

  // Extract the final max value
  uint8_t max_value = static_cast<uint8_t>(_mm_extract_epi8(max_128, 0));

  for (; i < size; ++i) {
    if (data[i] > max_value) {
      max_value = data[i];
    }
  }

  return max_value;
}

int8_t reduce_max_i8_avx2(const int8_t* data, size_t size) {
  __m256i max_vec = _mm256_set1_epi8(INT8_MIN);

  size_t i;
  for (i = 0; i + 32 <= size; i += 32) {
    __m256i vec = _mm256_loadu_si256((const __m256i*)(data + i));
    max_vec = _mm256_max_epi8(max_vec, vec);
  }

  int8_t remaining_max = INT8_MIN;
  for (; i < size; ++i) {
    if (data[i] > remaining_max) {
      remaining_max = data[i];
    }
  }

  // Now reduce the 256-bit max_vec into a single max value
  // First, split the 256-bit vector into two 128-bit halves and compute the max
  // between them
  __m128i max_128 = _mm_max_epi8(_mm256_castsi256_si128(max_vec), _mm256_extracti128_si256(max_vec, 1));

  // Further reduce the 128-bit vector to a scalar
  // Extract the upper 64-bit part and compute the max
  max_128 = _mm_max_epi8(max_128, _mm_srli_si128(max_128, 8));
  // Extract the upper 32-bit part and compute the max
  max_128 = _mm_max_epi8(max_128, _mm_srli_si128(max_128, 4));
  // Extract the upper 16-bit part and compute the max
  max_128 = _mm_max_epi8(max_128, _mm_srli_si128(max_128, 2));
  // Extract the upper 8-bit part and compute the max
  max_128 = _mm_max_epi8(max_128, _mm_srli_si128(max_128, 1));

  // Extract the final max value
  int8_t max_value = static_cast<int8_t>(_mm_extract_epi8(max_128, 0));

  if (remaining_max > max_value) {
    max_value = remaining_max;
  }

  return max_value;
}

MLAS_FORCEINLINE
__m128i MlasFloatToI8Avx2(const __m256 float_val1, const __m256 float_val2) {
  __m256 rounded_val1 = _mm256_round_ps(float_val1, _MM_FROUND_TO_NEAREST_INT);
  __m256 rounded_val2 = _mm256_round_ps(float_val2, _MM_FROUND_TO_NEAREST_INT);
  __m256i int_vec1 = _mm256_cvtps_epi32(rounded_val1);
  __m256i int_vec2 = _mm256_cvtps_epi32(rounded_val2);

  __m256i packed16_1 = _mm256_packs_epi32(int_vec1, int_vec2);
  __m128i packed8 = _mm_packs_epi16(_mm256_castsi256_si128(packed16_1), _mm256_extracti128_si256(packed16_1, 1));
  __m128i lanefix = _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(packed8), _mm_setr_epi32(0, 2, 1, 3)));
  return lanefix;
}

MLAS_FORCEINLINE
__m128i MlasFloatToU8Avx2(const __m256 float_val1, const __m256 float_val2) {
  __m256 rounded_val1 = _mm256_round_ps(float_val1, _MM_FROUND_TO_NEAREST_INT);
  __m256 rounded_val2 = _mm256_round_ps(float_val2, _MM_FROUND_TO_NEAREST_INT);
  __m256i int_vec1 = _mm256_cvtps_epi32(rounded_val1);
  __m256i int_vec2 = _mm256_cvtps_epi32(rounded_val2);

  __m256i packed16_1 = _mm256_packus_epi32(int_vec1, int_vec2);
  __m128i packed8 = _mm_packus_epi16(_mm256_castsi256_si128(packed16_1), _mm256_extracti128_si256(packed16_1, 1));
  __m128i lanefix = _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(packed8), _mm_setr_epi32(0, 2, 1, 3)));
  return lanefix;
}

float exp_and_sum_i8_avx2(const float* base_addr, const int8_t* indice, size_t size, int32_t adjustment,
                          float* temp_out) {
  __m256 sum = _mm256_setzero_ps();
  __m128i broadcast_adjustment = _mm_set1_epi8(static_cast<int8_t>(adjustment));
  //======================reduce sum start=========================
  size_t i;
  for (i = 0; i + 16 <= size; i += 16) {
    __m128i index_ori = _mm_loadu_si128((const __m128i*)(indice + i));
    __m128i index = _mm_add_epi8(index_ori, broadcast_adjustment);

    __m256i vec32_low = _mm256_cvtepu8_epi32(index);
    __m256 gathered = _mm256_i32gather_ps(base_addr, vec32_low, 4);
    sum = _mm256_add_ps(sum, gathered);
    _mm256_storeu_ps(&temp_out[i], gathered);

    __m128i vec8_high = _mm_srli_si128(index, 8);
    __m256i vec32_high = _mm256_cvtepu8_epi32(vec8_high);
    gathered = _mm256_i32gather_ps(base_addr, vec32_high, 4);
    sum = _mm256_add_ps(sum, gathered);
    _mm256_storeu_ps(&temp_out[i + 8], gathered);
  }
  float partial_sum = 0;
  for (; i < size; ++i) {
    float data = base_addr[uint8_t(indice[i] + adjustment)];
    partial_sum += data;
    temp_out[i] = data;
  }
  alignas(32) float results[8];
  _mm256_store_ps(results, sum);
  float total_sum = partial_sum;
  for (size_t j = 0; j < 8; ++j) {
    total_sum += results[j];
  }
  return total_sum;
}

float exp_and_sum_u8_avx2(const float* base_addr, const uint8_t* indice, size_t size, int32_t, float* temp_out) {
  __m256 sum = _mm256_setzero_ps();
  //======================reduce sum start=========================
  size_t i;
  for (i = 0; i + 16 <= size; i += 16) {
    __m128i index = _mm_loadu_si128((const __m128i*)(indice + i));
    __m256i vec32_low = _mm256_cvtepu8_epi32(index);
    __m256 gathered = _mm256_i32gather_ps(base_addr, vec32_low, 4);
    sum = _mm256_add_ps(sum, gathered);
    _mm256_storeu_ps(&temp_out[i], gathered);

    __m128i vec8_high = _mm_srli_si128(index, 8);
    __m256i vec32_high = _mm256_cvtepu8_epi32(vec8_high);
    gathered = _mm256_i32gather_ps(base_addr, vec32_high, 4);
    sum = _mm256_add_ps(sum, gathered);
    _mm256_storeu_ps(&temp_out[i + 8], gathered);
  }
  float partial_sum = 0;
  for (; i < size; ++i) {
    float data = base_addr[indice[i]];
    partial_sum += data;
    temp_out[i] = data;
  }
  alignas(32) float results[8];
  _mm256_store_ps(results, sum);
  float total_sum = partial_sum;
  for (size_t j = 0; j < 8; ++j) {
    total_sum += results[j];
  }
  return total_sum;
}

int32_t normalize_sum_avx2(float total_sum, size_t size, float x_scale, float* temp_out, float yzp, int8_t* output) {
  size_t i;

  //======================m scale d sum p zero start=========================
  float inverse_sum = 1.0f / total_sum;
  float scale = inverse_sum * x_scale;
  __m256 broadcast_scale = _mm256_broadcast_ss(&scale);
  __m256 broadcast_zp = _mm256_broadcast_ss(&yzp);

  // div sum
  for (i = 0; i + 16 <= size; i += 16) {
    __m256 vec1 = _mm256_loadu_ps(&temp_out[i]);
    __m256 product1 = _mm256_mul_ps(vec1, broadcast_scale);
    __m256 fma_result1 = _mm256_add_ps(product1, broadcast_zp);

    __m256 vec2 = _mm256_loadu_ps(&temp_out[i + 8]);
    __m256 product2 = _mm256_mul_ps(vec2, broadcast_scale);
    __m256 fma_result2 = _mm256_add_ps(product2, broadcast_zp);

    __m128i packed8 = MlasFloatToI8Avx2(fma_result1, fma_result2);

    _mm_storeu_si128((__m128i*)&output[i], packed8);
  }

  constexpr uint8_t max_u8 = 255;
  for (; i < size; ++i) {
    int v = int32_t(std::nearbyintf(temp_out[i] * scale + yzp));
    output[i] = v > max_u8 ? static_cast<int8_t>(max_u8) : static_cast<int8_t>(v);
  }

  return 0;
}

int32_t normalize_sum_avx2(float total_sum, size_t size, float x_scale, float* temp_out, float yzp, uint8_t* output) {
  size_t i;

  //======================m scale d sum p zero start=========================
  float inverse_sum = 1.0f / total_sum;
  float scale = inverse_sum * x_scale;
  __m256 broadcast_scale = _mm256_broadcast_ss(&scale);
  __m256 broadcast_zp = _mm256_broadcast_ss(&yzp);

  // div sum
  for (i = 0; i + 16 <= size; i += 16) {
    __m256 vec1 = _mm256_loadu_ps(&temp_out[i]);
    __m256 product1 = _mm256_mul_ps(vec1, broadcast_scale);
    __m256 fma_result1 = _mm256_add_ps(product1, broadcast_zp);

    __m256 vec2 = _mm256_loadu_ps(&temp_out[i + 8]);
    __m256 product2 = _mm256_mul_ps(vec2, broadcast_scale);
    __m256 fma_result2 = _mm256_add_ps(product2, broadcast_zp);

    __m128i packed8 = MlasFloatToU8Avx2(fma_result1, fma_result2);

    _mm_storeu_si128((__m128i*)&output[i], packed8);
  }
  constexpr uint8_t max_u8 = 255;
  for (; i < size; ++i) {
    int v = int32_t(std::nearbyintf(temp_out[i] * scale + yzp));
    output[i] = v > max_u8 ? max_u8 : static_cast<uint8_t>(v);
  }

  return 0;
}

// compute softmax for a row with D elements
void MlasQuantizeSoftmaxI8KernelAvx2(size_t D, const int8_t* x_data, int8_t* y_data, const float* lookup_table,
                                     float y_scale, int8_t yzp, float* tempaddr) {
  int32_t xmax = reduce_max_i8_avx2(x_data, D);
  const int32_t adjustment = int32_t(127) - xmax;
  const float* shifted_lookuptable = lookup_table;
  float total_sum = exp_and_sum_i8_avx2(shifted_lookuptable, x_data, D, adjustment, (float*)tempaddr);
  normalize_sum_avx2(total_sum, D, y_scale, (float*)tempaddr, yzp, y_data);
}

void MlasQuantizeSoftmaxU8KernelAvx2(size_t D, const uint8_t* x_data, uint8_t* y_data, const float* lookup_table,
                                     float y_scale, uint8_t yzp, float* tempaddr) {
  int32_t xmax = reduce_max_u8_avx2(x_data , D);
  const int32_t adjustment = int32_t(255) - xmax;
  const float* shifted_lookuptable = lookup_table + adjustment;
  float total_sum = exp_and_sum_u8_avx2(shifted_lookuptable, x_data, D, adjustment, (float*)tempaddr);
  normalize_sum_avx2(total_sum, D, y_scale, (float*)tempaddr, yzp, y_data);
}
