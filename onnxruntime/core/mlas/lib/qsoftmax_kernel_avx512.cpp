/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qsoftmax_kernel_avx512.cpp.h

Abstract:

    This module implements the lookup-based quantized softmax kernels for x64 avx512.

--*/
#include "mlas.h"
#include "mlasi.h"

uint8_t reduce_max_u8_avx512(const uint8_t* data, size_t size) {
  // Initialize max value to the smallest possible uint8_t (0)
  __m512i max_val = _mm512_set1_epi8(0);  // Set the initial max value to 0 for unsigned
  size_t i;
  // Process data in chunks of 64 bytes (512 bits, which is 64 * 8-bit integers)
  for (i = 0; i + 64 < size; i += 64) {
    // Load 64 bytes into a 512-bit register
    __m512i vec = _mm512_loadu_si512((__m512i*)&data[i]);

    // Compute the maximum values
    max_val = _mm512_max_epu8(max_val, vec);  // Use unsigned comparison
  }

  // Reduce the final max_val to find the maximum element
  // Extract the upper 256 bits and compare with the lower 256 bits
  __m256i max256 = _mm512_extracti64x4_epi64(max_val, 1);  // Extract upper 256 bits
  max256 = _mm256_max_epu8(max256,
                           _mm512_castsi512_si256(max_val));  // Compare upper 256 with lower 256

  // Further reduce 256-bit value
  __m128i max128 = _mm256_extracti128_si256(max256, 1);  // Extract upper 128 bits
  max128 = _mm_max_epu8(max128,
                        _mm256_castsi256_si128(max256));  // Compare upper 128 with lower 128

  // Further reduce 128-bit value
  max128 = _mm_max_epu8(max128, _mm_srli_si128(max128, 8));  // Compare first 8 bytes with second 8 bytes
  max128 = _mm_max_epu8(max128, _mm_srli_si128(max128, 4));  // Further reduce
  max128 = _mm_max_epu8(max128, _mm_srli_si128(max128, 2));  // Further reduce
  max128 = _mm_max_epu8(max128, _mm_srli_si128(max128, 1));  // Final reduce

  // The maximum value is now in the first byte of max128
  uint8_t max_value = static_cast<uint8_t>(_mm_extract_epi8(max128, 0));  // Extract the maximum value

  for (; i < size; ++i) {
    if (data[i] > max_value) {
      max_value = data[i];
    }
  }

  return max_value;
}

int8_t reduce_max_i8_avx512(const int8_t* data, size_t size) {
  size_t i;
  __m512i max_val = _mm512_set1_epi8(INT8_MIN);  // Start with the minimum signed value

  // Process data in chunks of 64 bytes (512 bits, which is 64 * 8-bit integers)
  for (i = 0; i + 64 < size; i += 64) {
    // Load 64 bytes into a 512-bit register
    __m512i vec = _mm512_loadu_si512((__m512i*)&data[i]);

    // Compute the maximum values
    max_val = _mm512_max_epi8(max_val, vec);
  }

  // Reduce the final max_val to find the maximum element
  // Extract the upper 256 bits and compare with the lower 256 bits
  __m256i max256 = _mm512_extracti64x4_epi64(max_val, 1);  // Extract upper 256 bits
  max256 = _mm256_max_epi8(max256,
                           _mm512_castsi512_si256(max_val));  // Compare upper 256 with lower 256

  // Further reduce 256-bit value
  __m128i max128 = _mm256_extracti128_si256(max256, 1);  // Extract upper 128 bits
  max128 = _mm_max_epi8(max128,
                        _mm256_castsi256_si128(max256));  // Compare upper 128 with lower 128

  // Further reduce 128-bit value
  max128 = _mm_max_epi8(max128, _mm_srli_si128(max128, 8));  // Compare first 8 bytes with second 8 bytes
  max128 = _mm_max_epi8(max128, _mm_srli_si128(max128, 4));  // Further reduce
  max128 = _mm_max_epi8(max128, _mm_srli_si128(max128, 2));  // Further reduce
  max128 = _mm_max_epi8(max128, _mm_srli_si128(max128, 1));  // Final reduce

  int8_t sc_max_value = static_cast<int8_t>(_mm_extract_epi8(max128, 0));
  for (; i < size; ++i) {
    if (data[i] > sc_max_value) {
      sc_max_value = data[i];
    }
  }

  return sc_max_value;
}

__m128i convert_float_to_u8_avx512bw(__m512 float_vals) {
  // Apply rounding
  __m512 rounded_vals = _mm512_roundscale_ps(float_vals, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

  // Convert float to int
  __m512i int_vals = _mm512_cvttps_epi32(rounded_vals);

  __m256i f256 = _mm512_extracti64x4_epi64(int_vals, 0);
  __m256i s256 = _mm512_extracti64x4_epi64(int_vals, 1);

  __m256i packed16_1 = _mm256_packus_epi32(f256, s256);
  __m128i packed8 = _mm_packus_epi16(_mm256_castsi256_si128(packed16_1), _mm256_extracti128_si256(packed16_1, 1));
  __m128i lanefix = _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(packed8), _mm_setr_epi32(0, 2, 1, 3)));
  return lanefix;
}

__m128i convert_float_to_i8_avx512bw(__m512 float_vals) {
  // Apply rounding
  __m512 rounded_vals = _mm512_roundscale_ps(float_vals, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

  // Convert float to int
  __m512i int_vals = _mm512_cvttps_epi32(rounded_vals);

  __m256i f256 = _mm512_extracti64x4_epi64(int_vals, 0);
  __m256i s256 = _mm512_extracti64x4_epi64(int_vals, 1);

  __m256i packed16_1 = _mm256_packs_epi32(f256, s256);
  __m128i packed8 = _mm_packs_epi16(_mm256_castsi256_si128(packed16_1), _mm256_extracti128_si256(packed16_1, 1));
  __m128i lanefix = _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(packed8), _mm_setr_epi32(0, 2, 1, 3)));
  return lanefix;
}

float exp_and_sum_i8_avx512(const float* base_addr, const int8_t* indice, size_t size, int32_t adjustment,
                            float* temp_out) {
  __m512 sum = _mm512_setzero_ps();
  __m256i broadcast_adjustment = _mm256_set1_epi8(static_cast<int8_t>(adjustment));

  size_t i = 0;
  for (; i + 32 <= size; i += 32) {
    __m256i index_ori = _mm256_loadu_si256((__m256i*)&indice[i]);
    __m256i index = _mm256_add_epi8(index_ori, broadcast_adjustment);
    // Extract the lower 128 bits (first half)
    __m128i index_low = _mm256_extracti128_si256(index, 0);
    __m512i vec32_low = _mm512_cvtepu8_epi32(index_low);
    __m512 gathered_data = _mm512_i32gather_ps(vec32_low, base_addr, 4);
    sum = _mm512_add_ps(sum, gathered_data);
    _mm512_storeu_ps(&temp_out[i], gathered_data);

    // Extract the upper 128 bits (second half)
    __m128i index_high = _mm256_extracti128_si256(index, 1);
    __m512i vec32_high = _mm512_cvtepu8_epi32(index_high);
    gathered_data = _mm512_i32gather_ps(vec32_high, base_addr, 4);
    sum = _mm512_add_ps(sum, gathered_data);
    _mm512_storeu_ps(&temp_out[i + 16], gathered_data);
  }
  // Reduce sum to a scalar value
  // Use shuffle and add to accumulate the result within the 512-bit register
  __m512 shuf = _mm512_shuffle_f32x4(sum, sum, 0b11110101);  // Swap 128-bit halves
  sum = _mm512_add_ps(sum, shuf);                            // Add swapped halves

  shuf = _mm512_shuffle_f32x4(sum, sum, 0b01001110);  // Further shuffle within 128-bit lanes
  sum = _mm512_add_ps(sum, shuf);                     // Add

  // Now reduce within the 128-bit lanes
  shuf = _mm512_shuffle_ps(sum, sum, 0b10110001);  // Swap pairs of elements
  sum = _mm512_add_ps(sum, shuf);                  // Add

  shuf = _mm512_shuffle_ps(sum, sum, 0b01001110);  // Further shuffle pairs
  sum = _mm512_add_ps(sum, shuf);                  // Add

  float total = _mm_cvtss_f32(_mm512_castps512_ps128(sum));
  for (; i < size; ++i) {
    float v = base_addr[uint8_t(indice[i] + adjustment)];
    temp_out[i] = v;
    total += v;
  }

  return total;
}

float exp_and_sum_u8_avx512(const float* base_addr, const uint8_t* indice, size_t size, int32_t, float* temp_out) {
  __m512 sum = _mm512_setzero_ps();

  size_t i = 0;
  for (; i + 32 <= size; i += 32) {
    __m256i index = _mm256_loadu_si256((__m256i*)&indice[i]);
    // Extract the lower 128 bits (first half)
    __m128i index_low = _mm256_extracti128_si256(index, 0);
    __m512i vec32_low = _mm512_cvtepu8_epi32(index_low);
    __m512 gathered_data = _mm512_i32gather_ps(vec32_low, base_addr, 4);
    sum = _mm512_add_ps(sum, gathered_data);
    _mm512_storeu_ps(&temp_out[i], gathered_data);

    // Extract the upper 128 bits (second half)
    __m128i index_high = _mm256_extracti128_si256(index, 1);
    __m512i vec32_high = _mm512_cvtepu8_epi32(index_high);
    gathered_data = _mm512_i32gather_ps(vec32_high, base_addr, 4);
    sum = _mm512_add_ps(sum, gathered_data);
    _mm512_storeu_ps(&temp_out[i + 16], gathered_data);
  }
  // Reduce sum to a scalar value
  // Use shuffle and add to accumulate the result within the 512-bit register
  __m512 shuf = _mm512_shuffle_f32x4(sum, sum, 0b11110101);  // Swap 128-bit halves
  sum = _mm512_add_ps(sum, shuf);                            // Add swapped halves

  shuf = _mm512_shuffle_f32x4(sum, sum, 0b01001110);  // Further shuffle within 128-bit lanes
  sum = _mm512_add_ps(sum, shuf);                     // Add

  // Now reduce within the 128-bit lanes
  shuf = _mm512_shuffle_ps(sum, sum, 0b10110001);  // Swap pairs of elements
  sum = _mm512_add_ps(sum, shuf);                  // Add

  shuf = _mm512_shuffle_ps(sum, sum, 0b01001110);  // Further shuffle pairs
  sum = _mm512_add_ps(sum, shuf);                  // Add

  float total = _mm_cvtss_f32(_mm512_castps512_ps128(sum));
  for (; i < size; ++i) {
    float v = base_addr[indice[i]];
    temp_out[i] = v;
    total += v;
  }

  return total;
}

int32_t normalize_sum_avx512(float total_sum, size_t size, float x_scale, float* temp_out, float yzp, uint8_t* output) {
  size_t i = 0;
  float inverse_sum = 1.0f / total_sum;
  float scale = inverse_sum * x_scale;
  __m512 broadcast_scale = _mm512_set1_ps(scale);
  __m512 broadcast_zp = _mm512_set1_ps(yzp);

  for (i = 0; i + 16 <= size; i += 16) {
    __m512 a_vec = _mm512_loadu_ps(&temp_out[i]);
    __m512 result_vec = _mm512_fmadd_ps(a_vec, broadcast_scale, broadcast_zp);
    // __m512i fma_result1 = _mm512_cvtt_roundps_epi32(result_vec,
    // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m128i packed8 = convert_float_to_u8_avx512bw(result_vec);
    _mm_storeu_si128((__m128i*)&output[i], packed8);
  }
  constexpr uint8_t max_u8 = 255;
  for (; i < size; ++i) {
    int v = int32_t(std::nearbyintf(temp_out[i] * scale + yzp));
    output[i] = v > max_u8 ? max_u8 : static_cast<uint8_t>(v);
  }
  return 0;
}

int32_t normalize_sum_avx512(float total_sum, size_t size, float x_scale, float* temp_out, float yzp, int8_t* output) {
  size_t i = 0;
  float inverse_sum = 1.0f / total_sum;
  float scale = inverse_sum * x_scale;
  __m512 broadcast_scale = _mm512_set1_ps(scale);
  __m512 broadcast_zp = _mm512_set1_ps(yzp);

  for (i = 0; i + 16 <= size; i += 16) {
    __m512 a_vec = _mm512_loadu_ps(&temp_out[i]);
    __m512 result_vec = _mm512_fmadd_ps(a_vec, broadcast_scale, broadcast_zp);
    // __m512i fma_result1 = _mm512_cvtt_roundps_epi32(result_vec,
    // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m128i packed8 = convert_float_to_i8_avx512bw(result_vec);
    _mm_storeu_si128((__m128i*)&output[i], packed8);
  }
  constexpr uint8_t max_u8 = 255;
  for (; i < size; ++i) {
    int v = int32_t(std::nearbyintf(temp_out[i] * scale + yzp));
    output[i] = v > max_u8 ? static_cast<int8_t>(max_u8) : static_cast<int8_t>(v);
  }
  return 0;
}

void MlasQuantizeSoftmaxI8KernelAvx512(size_t D, const int8_t* x_data, int8_t* y_data,
                                       const float* lookup_table, float y_scale, int8_t yzp, float* tempaddr) {
  int32_t xmax = reduce_max_i8_avx512(x_data, D);
  const int32_t adjustment = int32_t(127) - xmax;
  const float* shifted_lookuptable = lookup_table;
  float total_sum = exp_and_sum_i8_avx512(shifted_lookuptable, x_data, D, adjustment, (float*)tempaddr);
  normalize_sum_avx512(total_sum, D, y_scale, (float*)tempaddr, yzp, y_data);
}

void MlasQuantizeSoftmaxU8KernelAvx512(size_t D, const uint8_t* x_data, uint8_t* y_data,
                                       const float* lookup_table, float y_scale, uint8_t yzp, float* tempaddr) {
  int32_t xmax = reduce_max_u8_avx512(x_data, D);
  const int32_t adjustment = int32_t(255) - xmax;
  const float* shifted_lookuptable = lookup_table + adjustment;
  float total_sum = exp_and_sum_u8_avx512(shifted_lookuptable, x_data, D, adjustment, (float*)tempaddr);
  normalize_sum_avx512(total_sum, D, y_scale, (float*)tempaddr, yzp, y_data);
}
