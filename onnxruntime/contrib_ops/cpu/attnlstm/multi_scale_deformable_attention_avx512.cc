// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/attnlstm/multi_scale_deformable_attention.h"

#include <cstring>

#include <immintrin.h>

#define PREFETCH_ADDR_8x(ADDRESSES, HINT)   \
  { \
      __m256i addr_1sthalf = _mm512_extracti64x4_epi64(ADDRESSES, 0); \
      __m256i addr_2ndhalf = _mm512_extracti64x4_epi64(ADDRESSES, 1); \
      _mm_prefetch(reinterpret_cast<char *>(_mm256_extract_epi64(addr_1sthalf, 0)), HINT);    \
      _mm_prefetch(reinterpret_cast<char *>(_mm256_extract_epi64(addr_1sthalf, 1)), HINT);    \
      _mm_prefetch(reinterpret_cast<char *>(_mm256_extract_epi64(addr_1sthalf, 2)), HINT);    \
      _mm_prefetch(reinterpret_cast<char *>(_mm256_extract_epi64(addr_1sthalf, 3)), HINT);    \
      _mm_prefetch(reinterpret_cast<char *>(_mm256_extract_epi64(addr_2ndhalf, 0)), HINT);    \
      _mm_prefetch(reinterpret_cast<char *>(_mm256_extract_epi64(addr_2ndhalf, 1)), HINT);    \
      _mm_prefetch(reinterpret_cast<char *>(_mm256_extract_epi64(addr_2ndhalf, 2)), HINT);    \
      _mm_prefetch(reinterpret_cast<char *>(_mm256_extract_epi64(addr_2ndhalf, 3)), HINT);    \
  }

namespace onnxruntime {
namespace contrib {
  void MultiScaleDeformableAttention::ComputeAVX512(
    const float* value,
    const int64_t* value_spatial_shapes,
    const float* reference_points,
    const float* sampling_locations,
    const float* attention_weights,
    float* output,
    int64_t M,
    int64_t L,
    int64_t P,
    int64_t D,
    int64_t Q,
    concurrency::ThreadPool* thread_pool,
    AllocatorPtr alloc) const {
    constexpr size_t AVX512Alignment = 64;
    constexpr int64_t fp32step = static_cast<int64_t>(AVX512Alignment / sizeof(float));
    constexpr int64_t int64step = static_cast<int64_t>(AVX512Alignment / sizeof(int64_t));

    uint32_t threadCount = static_cast<uint32_t>(concurrency::ThreadPool::DegreeOfParallelism(thread_pool));
    if(Q <= 32){
        threadCount = 1;
    }

    std::vector<std::pair<int32_t, int32_t>> ranges(threadCount);

    const size_t perThreadWorkSize = M * P * (2 * sizeof(float) + 4 * sizeof(intptr_t));
    static_assert(sizeof(int64_t) == sizeof(intptr_t), "int64_t and intptr_t should have the same size");
    const size_t zeroRegionSize = D * sizeof(float);
    const size_t totalWorkSize = threadCount * perThreadWorkSize + zeroRegionSize;

    char * buffer = static_cast<char *>(alloc->AllocArrayWithAlignment<AVX512Alignment>(totalWorkSize, sizeof(char)));

    int64_t oobAddress = 0;
    {
        float * zeroRegion = reinterpret_cast<float *>(buffer + threadCount * perThreadWorkSize);
        memset(zeroRegion, 0, zeroRegionSize);
        oobAddress = reinterpret_cast<int64_t>(zeroRegion);
    }

    uint32_t rangeSize = static_cast<uint32_t>(Q) / threadCount;
    uint32_t remainder = static_cast<uint32_t>(Q) % threadCount;
    int32_t start = 0;
    for(uint32_t i = 0; i < remainder; ++i){
        int32_t end = start + rangeSize + 1;
        ranges[i].first = start;
        ranges[i].second = end;
        start = end;
    }
    for(uint32_t i = remainder; i < threadCount; ++i){
        int32_t end = start + rangeSize;
        ranges[i].first = start;
        ranges[i].second = end;
        start = end;
    }

    auto worker_lambda = [&](std::ptrdiff_t threadId) -> void {
      // Constants
      const __m512 zero = _mm512_setzero_ps();
      const __m512 one_f = _mm512_set1_ps(1.f);
      const __m512 half_f = _mm512_set1_ps(0.5f);
      const __m512i one_i = _mm512_set1_epi32(1);
      const __m512i zero_i64 = _mm512_setzero_epi32();
      const __m512i oobAddressVec = _mm512_set1_epi64(oobAddress);

      int64_t * addr_tl = reinterpret_cast<int64_t*>(buffer + threadId * perThreadWorkSize);
      int64_t * addr_tr = addr_tl + M * P;
      int64_t * addr_bl = addr_tr + M * P;
      int64_t * addr_br = addr_bl + M * P;
      float * diff_h_low = reinterpret_cast<float*>(addr_br + M * P);
      float * diff_w_low = diff_h_low + M * P;

      _mm_prefetch(reinterpret_cast<char *>(addr_tl), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<char *>(addr_tr), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<char *>(addr_bl), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<char *>(addr_br), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<char *>(addr_tl + fp32step), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<char *>(addr_tr + fp32step), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<char *>(addr_bl + fp32step), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<char *>(addr_br + fp32step), _MM_HINT_T0);

      const float * feature_map_begin = value;
      // Handle different levels one by one
      for(int64_t source_level = 0; source_level < L; ++source_level){
        int32_t feature_map_height_i32 = static_cast<int32_t>(value_spatial_shapes[source_level * 2]);
        int32_t feature_map_width_i32 = static_cast<int32_t>(value_spatial_shapes[source_level * 2 + 1]);
        const __m512i feature_map_limit_h = _mm512_set1_epi32(feature_map_height_i32);
        const __m512i feature_map_limit_w = _mm512_set1_epi32(feature_map_width_i32);

        __m512i address_wwise_step_i64 = _mm512_set1_epi64(static_cast<int64_t>(1 << (7 + 2)));
        __m512i address_hwise_step_i64 = _mm512_set1_epi64(static_cast<int64_t>(feature_map_width_i32));
        address_hwise_step_i64 = _mm512_slli_epi64(address_hwise_step_i64, 7 + 2);
        __m512i addr_base = _mm512_set1_epi64(reinterpret_cast<int64_t>(feature_map_begin));

        __m512i attention_head_addr_offset = _mm512_set1_epi64(D * sizeof(float));
        // Generate a vector {0, 0, 0, 0, 4D, 4D, 4D, 4D} (from bit 0 to bit 511)
        // The assumption is that P is always 4 so the address is moved by D * sizeof(float) for every P sampling locations.
        // This is a multi-head attention and each head is sampling on a separate copy of values.
        __m512i attention_head_addr_intravector_offset = _mm512_inserti64x4(attention_head_addr_offset, _mm256_setzero_si256(), 0);
        // Generate a vector {8D, 8D, 8D, 8D, 8D, 8D, 8D, 8D} (from bit 0 to bit 511)
        attention_head_addr_offset = _mm512_slli_epi64(attention_head_addr_offset, 1);

        for(int32_t iq = ranges[threadId].first; iq < ranges[threadId].second; ++iq){
          auto offset = (source_level * Q + iq) * M * P * 2;

          // Load reference point for the current query
          float reference_point_h = reference_points[(source_level * Q + iq) * 2 + 1];
          float reference_point_w = reference_points[(source_level * Q + iq) * 2];
          __m512 reference_point_h_vec = _mm512_set1_ps(reference_point_h);
          __m512 reference_point_w_vec = _mm512_set1_ps(reference_point_w);

          // Lowest address placed in 0:31 because x86-64 is little-endian
          __m512 loc_wh_1stquarter = _mm512_loadu_ps(sampling_locations + offset);
          __m512 loc_wh_2ndquarter = _mm512_loadu_ps(sampling_locations + offset + fp32step);
          __m512 loc_wh_3rdquarter = _mm512_loadu_ps(sampling_locations + offset + 2 * fp32step);
          __m512 loc_wh_4thquarter = _mm512_loadu_ps(sampling_locations + offset + 3 * fp32step);

          __mmask16 mask_select_lo = _cvtu32_mask16(0x5555);
          __mmask16 mask_select_hi = _cvtu32_mask16(0xAAAA);

          // from bits[0:31] to bits[480:511] -- w0, 0, w1, 0, w2, 0, w3, 0, w4, 0, w5, 0, w6, 0, w7, 0
          __m512 loc_w_1stquarter = _mm512_mask_blend_ps(mask_select_lo, zero, loc_wh_1stquarter);
          __m512 loc_h_1stquarter = _mm512_mask_blend_ps(mask_select_hi, zero, loc_wh_1stquarter);
          __m512 loc_w_2ndquarter = _mm512_mask_blend_ps(mask_select_lo, zero, loc_wh_2ndquarter);
          __m512 loc_h_2ndquarter = _mm512_mask_blend_ps(mask_select_hi, zero, loc_wh_2ndquarter);
          __m512 loc_w_3rdquarter = _mm512_mask_blend_ps(mask_select_lo, zero, loc_wh_3rdquarter);
          __m512 loc_h_3rdquarter = _mm512_mask_blend_ps(mask_select_hi, zero, loc_wh_3rdquarter);
          __m512 loc_w_4thquarter = _mm512_mask_blend_ps(mask_select_lo, zero, loc_wh_4thquarter);
          __m512 loc_h_4thquarter = _mm512_mask_blend_ps(mask_select_hi, zero, loc_wh_4thquarter);

          __m256 loc_w_1stquarter_compressed = _mm256_castsi256_ps(_mm512_cvtepi64_epi32(_mm512_castps_si512(loc_w_1stquarter)));
          __m256 loc_w_2ndquarter_compressed = _mm256_castsi256_ps(_mm512_cvtepi64_epi32(_mm512_castps_si512(loc_w_2ndquarter)));
          __m512 loc_w_1sthalf = _mm512_insertf32x8(_mm512_castps256_ps512(loc_w_1stquarter_compressed), loc_w_2ndquarter_compressed, 1);

          __m256 loc_w_3rdquarter_compressed = _mm256_castsi256_ps(_mm512_cvtepi64_epi32(_mm512_castps_si512(loc_w_3rdquarter)));
          __m256 loc_w_4thquarter_compressed = _mm256_castsi256_ps(_mm512_cvtepi64_epi32(_mm512_castps_si512(loc_w_4thquarter)));
          __m512 loc_w_2ndhalf = _mm512_insertf32x8(_mm512_castps256_ps512(loc_w_3rdquarter_compressed), loc_w_4thquarter_compressed, 1);

          __m256 loc_h_1stquarter_compressed = _mm256_castsi256_ps(_mm512_cvtepi64_epi32(_mm512_ror_epi64(_mm512_castps_si512(loc_h_1stquarter), 32)));
          __m256 loc_h_2ndquarter_compressed = _mm256_castsi256_ps(_mm512_cvtepi64_epi32(_mm512_ror_epi64(_mm512_castps_si512(loc_h_2ndquarter), 32)));
          __m512 loc_h_1sthalf = _mm512_insertf32x8(_mm512_castps256_ps512(loc_h_1stquarter_compressed), loc_h_2ndquarter_compressed, 1);

          __m256 loc_h_3rdquarter_compressed = _mm256_castsi256_ps(_mm512_cvtepi64_epi32(_mm512_ror_epi64(_mm512_castps_si512(loc_h_3rdquarter), 32)));
          __m256 loc_h_4thquarter_compressed = _mm256_castsi256_ps(_mm512_cvtepi64_epi32(_mm512_ror_epi64(_mm512_castps_si512(loc_h_4thquarter), 32)));
          __m512 loc_h_2ndhalf = _mm512_insertf32x8(_mm512_castps256_ps512(loc_h_3rdquarter_compressed), loc_h_4thquarter_compressed, 1);

          // align_corners = False
          // [0, H] => [-0.5, H - 0.5]
          __m512 loc_h_scaled_1sthalf = loc_h_1sthalf;    // Scaling is removed
          loc_h_scaled_1sthalf = _mm512_add_ps(loc_h_scaled_1sthalf, reference_point_h_vec);
          loc_h_scaled_1sthalf = _mm512_sub_ps(loc_h_scaled_1sthalf, half_f);
          __m512 loc_h_scaled_2ndhalf = loc_h_2ndhalf;    // Scaling is removed
          loc_h_scaled_2ndhalf = _mm512_add_ps(loc_h_scaled_2ndhalf, reference_point_h_vec);
          loc_h_scaled_2ndhalf = _mm512_sub_ps(loc_h_scaled_2ndhalf, half_f);

          // align_corners = False
          // [0, W] => [-0.5, W - 0.5]
          __m512 loc_w_scaled_1sthalf = loc_w_1sthalf;    // Scaling is removed
          loc_w_scaled_1sthalf = _mm512_add_ps(loc_w_scaled_1sthalf, reference_point_w_vec);
          loc_w_scaled_1sthalf = _mm512_sub_ps(loc_w_scaled_1sthalf, half_f);
          __m512 loc_w_scaled_2ndhalf = loc_w_2ndhalf;    // Scaling is removed
          loc_w_scaled_2ndhalf = _mm512_add_ps(loc_w_scaled_2ndhalf, reference_point_w_vec);
          loc_w_scaled_2ndhalf = _mm512_sub_ps(loc_w_scaled_2ndhalf, half_f);

          __m512i loc_h_floor_1sthalf = _mm512_cvt_roundps_epi32(loc_h_scaled_1sthalf, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
          __m512i loc_h_floor_2ndhalf = _mm512_cvt_roundps_epi32(loc_h_scaled_2ndhalf, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
          __m512i loc_h_ceil_1sthalf = _mm512_add_epi32(loc_h_floor_1sthalf, one_i);
          __m512i loc_h_ceil_2ndhalf = _mm512_add_epi32(loc_h_floor_2ndhalf, one_i);
          __m512i loc_w_floor_1sthalf = _mm512_cvt_roundps_epi32(loc_w_scaled_1sthalf, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
          __m512i loc_w_floor_2ndhalf = _mm512_cvt_roundps_epi32(loc_w_scaled_2ndhalf, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
          __m512i loc_w_ceil_1sthalf = _mm512_add_epi32(loc_w_floor_1sthalf, one_i);
          __m512i loc_w_ceil_2ndhalf = _mm512_add_epi32(loc_w_floor_2ndhalf, one_i);

          __mmask16 valid_h_floor_1sthalf = _mm512_cmp_epi32_mask(loc_h_floor_1sthalf, zero_i64, _MM_CMPINT_GE);
          valid_h_floor_1sthalf = _mm512_kand(valid_h_floor_1sthalf, _mm512_cmp_epi32_mask(loc_h_floor_1sthalf, feature_map_limit_h, _MM_CMPINT_LT));
          __mmask16 valid_h_ceil_1sthalf = _mm512_cmp_epi32_mask(loc_h_ceil_1sthalf, zero_i64, _MM_CMPINT_GE);
          valid_h_ceil_1sthalf = _mm512_kand(valid_h_ceil_1sthalf, _mm512_cmp_epi32_mask(loc_h_ceil_1sthalf, feature_map_limit_h, _MM_CMPINT_LT));
          __mmask16 valid_w_floor_1sthalf = _mm512_cmp_epi32_mask(loc_w_floor_1sthalf, zero_i64, _MM_CMPINT_GE);
          valid_w_floor_1sthalf = _mm512_kand(valid_w_floor_1sthalf, _mm512_cmp_epi32_mask(loc_w_floor_1sthalf, feature_map_limit_w, _MM_CMPINT_LT));
          __mmask16 valid_w_ceil_1sthalf = _mm512_cmp_epi32_mask(loc_w_ceil_1sthalf, zero_i64, _MM_CMPINT_GE);
          valid_w_ceil_1sthalf = _mm512_kand(valid_w_ceil_1sthalf, _mm512_cmp_epi32_mask(loc_w_ceil_1sthalf, feature_map_limit_w, _MM_CMPINT_LT));

          __mmask16 valid_coord_tl_1sthalf = _mm512_kand(valid_h_floor_1sthalf, valid_w_floor_1sthalf);
          __mmask16 valid_coord_tr_1sthalf = _mm512_kand(valid_h_floor_1sthalf, valid_w_ceil_1sthalf);
          __mmask16 valid_coord_bl_1sthalf = _mm512_kand(valid_h_ceil_1sthalf, valid_w_floor_1sthalf);
          __mmask16 valid_coord_br_1sthalf = _mm512_kand(valid_h_ceil_1sthalf, valid_w_ceil_1sthalf);

          unsigned int mask_tl_1sthalf = _cvtmask16_u32(valid_coord_tl_1sthalf);
          unsigned int mask_tr_1sthalf = _cvtmask16_u32(valid_coord_tr_1sthalf);
          unsigned int mask_bl_1sthalf = _cvtmask16_u32(valid_coord_bl_1sthalf);
          unsigned int mask_br_1sthalf = _cvtmask16_u32(valid_coord_br_1sthalf);

          __mmask16 valid_h_floor_2ndhalf = _mm512_cmp_epi32_mask(loc_h_floor_2ndhalf, zero_i64, _MM_CMPINT_GE);
          valid_h_floor_2ndhalf = _mm512_kand(valid_h_floor_2ndhalf, _mm512_cmp_epi32_mask(loc_h_floor_2ndhalf, feature_map_limit_h, _MM_CMPINT_LT));
          __mmask16 valid_h_ceil_2ndhalf = _mm512_cmp_epi32_mask(loc_h_ceil_2ndhalf, zero_i64, _MM_CMPINT_GE);
          valid_h_ceil_2ndhalf = _mm512_kand(valid_h_ceil_2ndhalf, _mm512_cmp_epi32_mask(loc_h_ceil_2ndhalf, feature_map_limit_h, _MM_CMPINT_LT));
          __mmask16 valid_w_floor_2ndhalf = _mm512_cmp_epi32_mask(loc_w_floor_2ndhalf, zero_i64, _MM_CMPINT_GE);
          valid_w_floor_2ndhalf = _mm512_kand(valid_w_floor_2ndhalf, _mm512_cmp_epi32_mask(loc_w_floor_2ndhalf, feature_map_limit_w, _MM_CMPINT_LT));
          __mmask16 valid_w_ceil_2ndhalf = _mm512_cmp_epi32_mask(loc_w_ceil_2ndhalf, zero_i64, _MM_CMPINT_GE);
          valid_w_ceil_2ndhalf = _mm512_kand(valid_w_ceil_2ndhalf, _mm512_cmp_epi32_mask(loc_w_ceil_2ndhalf, feature_map_limit_w, _MM_CMPINT_LT));

          __mmask16 valid_coord_tl_2ndhalf = _mm512_kand(valid_h_floor_2ndhalf, valid_w_floor_2ndhalf);
          __mmask16 valid_coord_tr_2ndhalf = _mm512_kand(valid_h_floor_2ndhalf, valid_w_ceil_2ndhalf);
          __mmask16 valid_coord_bl_2ndhalf = _mm512_kand(valid_h_ceil_2ndhalf, valid_w_floor_2ndhalf);
          __mmask16 valid_coord_br_2ndhalf = _mm512_kand(valid_h_ceil_2ndhalf, valid_w_ceil_2ndhalf);

          unsigned int mask_tl_2ndhalf = _cvtmask16_u32(valid_coord_tl_2ndhalf);
          unsigned int mask_tr_2ndhalf = _cvtmask16_u32(valid_coord_tr_2ndhalf);
          unsigned int mask_bl_2ndhalf = _cvtmask16_u32(valid_coord_bl_2ndhalf);
          unsigned int mask_br_2ndhalf = _cvtmask16_u32(valid_coord_br_2ndhalf);

          __m512i addr_tl_1stquarter = addr_base;
          __m512i addr_tr_1stquarter = _mm512_add_epi64(addr_tl_1stquarter, address_wwise_step_i64);
          __m512i addr_bl_1stquarter = _mm512_add_epi64(addr_tl_1stquarter, address_hwise_step_i64);
          __m512i addr_br_1stquarter = _mm512_add_epi64(addr_bl_1stquarter, address_wwise_step_i64);
          __m512i addr_tl_2ndquarter = addr_base;
          __m512i addr_tr_2ndquarter = _mm512_add_epi64(addr_tl_2ndquarter, address_wwise_step_i64);
          __m512i addr_bl_2ndquarter = _mm512_add_epi64(addr_tl_2ndquarter, address_hwise_step_i64);
          __m512i addr_br_2ndquarter = _mm512_add_epi64(addr_bl_2ndquarter, address_wwise_step_i64);

          // Account for per-head offsets
          addr_tl_1stquarter = _mm512_add_epi64(addr_tl_1stquarter, attention_head_addr_intravector_offset);
          addr_tr_1stquarter = _mm512_add_epi64(addr_tr_1stquarter, attention_head_addr_intravector_offset);
          addr_bl_1stquarter = _mm512_add_epi64(addr_bl_1stquarter, attention_head_addr_intravector_offset);
          addr_br_1stquarter = _mm512_add_epi64(addr_br_1stquarter, attention_head_addr_intravector_offset);

          addr_tl_2ndquarter = _mm512_add_epi64(addr_tl_2ndquarter, attention_head_addr_offset);
          addr_tr_2ndquarter = _mm512_add_epi64(addr_tr_2ndquarter, attention_head_addr_offset);
          addr_bl_2ndquarter = _mm512_add_epi64(addr_bl_2ndquarter, attention_head_addr_offset);
          addr_br_2ndquarter = _mm512_add_epi64(addr_br_2ndquarter, attention_head_addr_offset);
          addr_tl_2ndquarter = _mm512_add_epi64(addr_tl_2ndquarter, attention_head_addr_intravector_offset);
          addr_tr_2ndquarter = _mm512_add_epi64(addr_tr_2ndquarter, attention_head_addr_intravector_offset);
          addr_bl_2ndquarter = _mm512_add_epi64(addr_bl_2ndquarter, attention_head_addr_intravector_offset);
          addr_br_2ndquarter = _mm512_add_epi64(addr_br_2ndquarter, attention_head_addr_intravector_offset);

          __m512i addr_offset_h_1stquarter = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(loc_h_floor_1sthalf, 0));
          __m512i addr_offset_h_2ndquarter = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(loc_h_floor_1sthalf, 1));
          __m512i addr_offset_1stquarter = _mm512_mul_epi32(addr_offset_h_1stquarter, feature_map_limit_w);
          __m512i addr_offset_2ndquarter = _mm512_mul_epi32(addr_offset_h_2ndquarter, feature_map_limit_w);
          __m512i addr_offset_w_1stquarter = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(loc_w_floor_1sthalf, 0));
          __m512i addr_offset_w_2ndquarter = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(loc_w_floor_1sthalf, 1));
          addr_offset_1stquarter = _mm512_add_epi64(addr_offset_1stquarter, addr_offset_w_1stquarter);
          addr_offset_2ndquarter = _mm512_add_epi64(addr_offset_2ndquarter, addr_offset_w_2ndquarter);

          // address_scale_factor = M * D = 128 = 2^7, sizeof(float) = 4 = 2^2
          addr_offset_1stquarter = _mm512_slli_epi64(addr_offset_1stquarter, 7 + 2);
          addr_offset_2ndquarter = _mm512_slli_epi64(addr_offset_2ndquarter, 7 + 2);

          addr_tl_1stquarter = _mm512_add_epi64(addr_tl_1stquarter, addr_offset_1stquarter);
          addr_tr_1stquarter = _mm512_add_epi64(addr_tr_1stquarter, addr_offset_1stquarter);
          addr_bl_1stquarter = _mm512_add_epi64(addr_bl_1stquarter, addr_offset_1stquarter);
          addr_br_1stquarter = _mm512_add_epi64(addr_br_1stquarter, addr_offset_1stquarter);
          addr_tl_2ndquarter = _mm512_add_epi64(addr_tl_2ndquarter, addr_offset_2ndquarter);
          addr_tr_2ndquarter = _mm512_add_epi64(addr_tr_2ndquarter, addr_offset_2ndquarter);
          addr_bl_2ndquarter = _mm512_add_epi64(addr_bl_2ndquarter, addr_offset_2ndquarter);
          addr_br_2ndquarter = _mm512_add_epi64(addr_br_2ndquarter, addr_offset_2ndquarter);

          __mmask8 valid_coord_tl_1stquarter = _cvtu32_mask8(mask_tl_1sthalf);
          __mmask8 valid_coord_tr_1stquarter = _cvtu32_mask8(mask_tr_1sthalf);
          __mmask8 valid_coord_bl_1stquarter = _cvtu32_mask8(mask_bl_1sthalf);
          __mmask8 valid_coord_br_1stquarter = _cvtu32_mask8(mask_br_1sthalf);

          addr_tl_1stquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_tl_1stquarter, addr_tl_1stquarter);
          addr_tr_1stquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_tr_1stquarter, addr_tr_1stquarter);
          addr_bl_1stquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_bl_1stquarter, addr_bl_1stquarter);
          addr_br_1stquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_br_1stquarter, addr_br_1stquarter);

          PREFETCH_ADDR_8x(addr_tl_1stquarter, _MM_HINT_T0);
          PREFETCH_ADDR_8x(addr_tr_1stquarter, _MM_HINT_T0);
          PREFETCH_ADDR_8x(addr_bl_1stquarter, _MM_HINT_T0);
          PREFETCH_ADDR_8x(addr_br_1stquarter, _MM_HINT_T0);

          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_tl), addr_tl_1stquarter);
          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_tr), addr_tr_1stquarter);
          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_bl), addr_bl_1stquarter);
          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_br), addr_br_1stquarter);

          __mmask8 valid_coord_tl_2ndquarter = _cvtu32_mask8(mask_tl_1sthalf >> 8);
          __mmask8 valid_coord_tr_2ndquarter = _cvtu32_mask8(mask_tr_1sthalf >> 8);
          __mmask8 valid_coord_bl_2ndquarter = _cvtu32_mask8(mask_bl_1sthalf >> 8);
          __mmask8 valid_coord_br_2ndquarter = _cvtu32_mask8(mask_br_1sthalf >> 8);

          addr_tl_2ndquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_tl_2ndquarter, addr_tl_2ndquarter);
          addr_tr_2ndquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_tr_2ndquarter, addr_tr_2ndquarter);
          addr_bl_2ndquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_bl_2ndquarter, addr_bl_2ndquarter);
          addr_br_2ndquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_br_2ndquarter, addr_br_2ndquarter);

          PREFETCH_ADDR_8x(addr_tl_2ndquarter, _MM_HINT_T1);
          PREFETCH_ADDR_8x(addr_tr_2ndquarter, _MM_HINT_T1);
          PREFETCH_ADDR_8x(addr_bl_2ndquarter, _MM_HINT_T1);
          PREFETCH_ADDR_8x(addr_br_2ndquarter, _MM_HINT_T1);

          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_tl) + int64step, addr_tl_2ndquarter);
          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_tr) + int64step, addr_tr_2ndquarter);
          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_bl) + int64step, addr_bl_2ndquarter);
          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_br) + int64step, addr_br_2ndquarter);

          __m512i addr_tl_3rdquarter = addr_base;
          __m512i addr_tr_3rdquarter = _mm512_add_epi64(addr_tl_3rdquarter, address_wwise_step_i64);
          __m512i addr_bl_3rdquarter = _mm512_add_epi64(addr_tl_3rdquarter, address_hwise_step_i64);
          __m512i addr_br_3rdquarter = _mm512_add_epi64(addr_bl_3rdquarter, address_wwise_step_i64);
          __m512i addr_tl_4thquarter = addr_base;
          __m512i addr_tr_4thquarter = _mm512_add_epi64(addr_tl_4thquarter, address_wwise_step_i64);
          __m512i addr_bl_4thquarter = _mm512_add_epi64(addr_tl_4thquarter, address_hwise_step_i64);
          __m512i addr_br_4thquarter = _mm512_add_epi64(addr_bl_4thquarter, address_wwise_step_i64);

          // Account for per-head offsets
          addr_tl_3rdquarter = _mm512_add_epi64(addr_tl_3rdquarter, _mm512_slli_epi64(attention_head_addr_offset, 1));
          addr_tr_3rdquarter = _mm512_add_epi64(addr_tr_3rdquarter, _mm512_slli_epi64(attention_head_addr_offset, 1));
          addr_bl_3rdquarter = _mm512_add_epi64(addr_bl_3rdquarter, _mm512_slli_epi64(attention_head_addr_offset, 1));
          addr_br_3rdquarter = _mm512_add_epi64(addr_br_3rdquarter, _mm512_slli_epi64(attention_head_addr_offset, 1));
          addr_tl_3rdquarter = _mm512_add_epi64(addr_tl_3rdquarter, attention_head_addr_intravector_offset);
          addr_tr_3rdquarter = _mm512_add_epi64(addr_tr_3rdquarter, attention_head_addr_intravector_offset);
          addr_bl_3rdquarter = _mm512_add_epi64(addr_bl_3rdquarter, attention_head_addr_intravector_offset);
          addr_br_3rdquarter = _mm512_add_epi64(addr_br_3rdquarter, attention_head_addr_intravector_offset);

          addr_tl_4thquarter = _mm512_add_epi64(addr_tl_4thquarter, _mm512_slli_epi64(attention_head_addr_offset, 1));
          addr_tr_4thquarter = _mm512_add_epi64(addr_tr_4thquarter, _mm512_slli_epi64(attention_head_addr_offset, 1));
          addr_bl_4thquarter = _mm512_add_epi64(addr_bl_4thquarter, _mm512_slli_epi64(attention_head_addr_offset, 1));
          addr_br_4thquarter = _mm512_add_epi64(addr_br_4thquarter, _mm512_slli_epi64(attention_head_addr_offset, 1));
          addr_tl_4thquarter = _mm512_add_epi64(addr_tl_4thquarter, attention_head_addr_offset);
          addr_tr_4thquarter = _mm512_add_epi64(addr_tr_4thquarter, attention_head_addr_offset);
          addr_bl_4thquarter = _mm512_add_epi64(addr_bl_4thquarter, attention_head_addr_offset);
          addr_br_4thquarter = _mm512_add_epi64(addr_br_4thquarter, attention_head_addr_offset);
          addr_tl_4thquarter = _mm512_add_epi64(addr_tl_4thquarter, attention_head_addr_intravector_offset);
          addr_tr_4thquarter = _mm512_add_epi64(addr_tr_4thquarter, attention_head_addr_intravector_offset);
          addr_bl_4thquarter = _mm512_add_epi64(addr_bl_4thquarter, attention_head_addr_intravector_offset);
          addr_br_4thquarter = _mm512_add_epi64(addr_br_4thquarter, attention_head_addr_intravector_offset);

          __m512i addr_offset_h_3rdquarter = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(loc_h_floor_2ndhalf, 0));
          __m512i addr_offset_h_4thquarter = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(loc_h_floor_2ndhalf, 1));
          __m512i addr_offset_3rdquarter = _mm512_mul_epi32(addr_offset_h_3rdquarter, feature_map_limit_w);
          __m512i addr_offset_4thquarter = _mm512_mul_epi32(addr_offset_h_4thquarter, feature_map_limit_w);
          __m512i addr_offset_w_3rdquarter = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(loc_w_floor_2ndhalf, 0));
          __m512i addr_offset_w_4thquarter = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(loc_w_floor_2ndhalf, 1));
          addr_offset_3rdquarter = _mm512_add_epi64(addr_offset_3rdquarter, addr_offset_w_3rdquarter);
          addr_offset_4thquarter = _mm512_add_epi64(addr_offset_4thquarter, addr_offset_w_4thquarter);

          // address_scale_factor = M * D = 128 = 2^7, sizeof(float) = 4 = 2^2
          addr_offset_3rdquarter = _mm512_slli_epi64(addr_offset_3rdquarter, 7 + 2);
          addr_offset_4thquarter = _mm512_slli_epi64(addr_offset_4thquarter, 7 + 2);

          addr_tl_3rdquarter = _mm512_add_epi64(addr_tl_3rdquarter, addr_offset_3rdquarter);
          addr_tr_3rdquarter = _mm512_add_epi64(addr_tr_3rdquarter, addr_offset_3rdquarter);
          addr_bl_3rdquarter = _mm512_add_epi64(addr_bl_3rdquarter, addr_offset_3rdquarter);
          addr_br_3rdquarter = _mm512_add_epi64(addr_br_3rdquarter, addr_offset_3rdquarter);
          addr_tl_4thquarter = _mm512_add_epi64(addr_tl_4thquarter, addr_offset_4thquarter);
          addr_tr_4thquarter = _mm512_add_epi64(addr_tr_4thquarter, addr_offset_4thquarter);
          addr_bl_4thquarter = _mm512_add_epi64(addr_bl_4thquarter, addr_offset_4thquarter);
          addr_br_4thquarter = _mm512_add_epi64(addr_br_4thquarter, addr_offset_4thquarter);

          __mmask8 valid_coord_tl_3rdquarter = _cvtu32_mask8(mask_tl_2ndhalf);
          __mmask8 valid_coord_tr_3rdquarter = _cvtu32_mask8(mask_tr_2ndhalf);
          __mmask8 valid_coord_bl_3rdquarter = _cvtu32_mask8(mask_bl_2ndhalf);
          __mmask8 valid_coord_br_3rdquarter = _cvtu32_mask8(mask_br_2ndhalf);

          addr_tl_3rdquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_tl_3rdquarter, addr_tl_3rdquarter);
          addr_tr_3rdquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_tr_3rdquarter, addr_tr_3rdquarter);
          addr_bl_3rdquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_bl_3rdquarter, addr_bl_3rdquarter);
          addr_br_3rdquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_br_3rdquarter, addr_br_3rdquarter);

          PREFETCH_ADDR_8x(addr_tl_3rdquarter, _MM_HINT_T1);
          PREFETCH_ADDR_8x(addr_tr_3rdquarter, _MM_HINT_T1);
          PREFETCH_ADDR_8x(addr_bl_3rdquarter, _MM_HINT_T1);
          PREFETCH_ADDR_8x(addr_br_3rdquarter, _MM_HINT_T1);

          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_tl) + 2 * int64step, addr_tl_3rdquarter);
          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_tr) + 2 * int64step, addr_tr_3rdquarter);
          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_bl) + 2 * int64step, addr_bl_3rdquarter);
          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_br) + 2 * int64step, addr_br_3rdquarter);

          __mmask8 valid_coord_tl_4thquarter = _cvtu32_mask8(mask_tl_2ndhalf >> 8);
          __mmask8 valid_coord_tr_4thquarter = _cvtu32_mask8(mask_tr_2ndhalf >> 8);
          __mmask8 valid_coord_bl_4thquarter = _cvtu32_mask8(mask_bl_2ndhalf >> 8);
          __mmask8 valid_coord_br_4thquarter = _cvtu32_mask8(mask_br_2ndhalf >> 8);

          addr_tl_4thquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_tl_4thquarter, addr_tl_4thquarter);
          addr_tr_4thquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_tr_4thquarter, addr_tr_4thquarter);
          addr_bl_4thquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_bl_4thquarter, addr_bl_4thquarter);
          addr_br_4thquarter = _mm512_mask_mov_epi64(oobAddressVec, valid_coord_br_4thquarter, addr_br_4thquarter);

          PREFETCH_ADDR_8x(addr_tl_4thquarter, _MM_HINT_T1);
          PREFETCH_ADDR_8x(addr_tr_4thquarter, _MM_HINT_T1);
          PREFETCH_ADDR_8x(addr_bl_4thquarter, _MM_HINT_T1);
          PREFETCH_ADDR_8x(addr_br_4thquarter, _MM_HINT_T1);

          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_tl) + 3 * int64step, addr_tl_4thquarter);
          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_tr) + 3 * int64step, addr_tr_4thquarter);
          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_bl) + 3 * int64step, addr_bl_4thquarter);
          _mm512_store_epi64(reinterpret_cast<int64_t*>(addr_br) + 3 * int64step, addr_br_4thquarter);

          __m512 loc_h_floor_1sthalf_f = _mm512_cvtepi32_ps(loc_h_floor_1sthalf);
          __m512 loc_h_floor_2ndhalf_f = _mm512_cvtepi32_ps(loc_h_floor_2ndhalf);
          __m512 loc_w_floor_1sthalf_f = _mm512_cvtepi32_ps(loc_w_floor_1sthalf);
          __m512 loc_w_floor_2ndhalf_f = _mm512_cvtepi32_ps(loc_w_floor_2ndhalf);

          __m512 coeff_h_low_1sthalf = _mm512_sub_ps(loc_h_scaled_1sthalf, loc_h_floor_1sthalf_f);
          __m512 coeff_h_low_2ndhalf = _mm512_sub_ps(loc_h_scaled_2ndhalf, loc_h_floor_2ndhalf_f);
          __m512 coeff_w_low_1sthalf = _mm512_sub_ps(loc_w_scaled_1sthalf, loc_w_floor_1sthalf_f);
          __m512 coeff_w_low_2ndhalf = _mm512_sub_ps(loc_w_scaled_2ndhalf, loc_w_floor_2ndhalf_f);

          _mm512_store_ps(diff_h_low, coeff_h_low_1sthalf);
          _mm512_store_ps(diff_h_low + fp32step, coeff_h_low_2ndhalf);
          _mm512_store_ps(diff_w_low, coeff_w_low_1sthalf);
          _mm512_store_ps(diff_w_low + fp32step, coeff_w_low_2ndhalf);

          _mm_prefetch(reinterpret_cast<char *>(addr_tl), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<char *>(addr_tr), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<char *>(addr_bl), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<char *>(addr_br), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<char *>(addr_tl + fp32step), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<char *>(addr_tr + fp32step), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<char *>(addr_bl + fp32step), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<char *>(addr_br + fp32step), _MM_HINT_T0);

          // location_index iterates from 0 to M * P (never reaches M * P)
          int64_t location_index = 0;
          for(int64_t im = 0; im < M; ++im)
          {
              int64_t output_index = iq * M * D + im * D;

              __m512 accumulator = zero;
              if(source_level > 0)
              {
                  accumulator = _mm512_loadu_ps(output + output_index);    // load value from previous source level
              }

              for(int64_t ip = 0; ip < P; ++ip)
              {
                  const int64_t weight_index = (offset >> 1) + location_index;
                  __m512 weight = _mm512_set1_ps(attention_weights[weight_index]);

                  __m512 data_tl = _mm512_loadu_ps(reinterpret_cast<float*>(addr_tl[location_index]));
                  __m512 data_tr = _mm512_loadu_ps(reinterpret_cast<float*>(addr_tr[location_index]));
                  __m512 data_bl = _mm512_loadu_ps(reinterpret_cast<float*>(addr_bl[location_index]));
                  __m512 data_br = _mm512_loadu_ps(reinterpret_cast<float*>(addr_br[location_index]));

                  __m512 coeff1 = _mm512_set1_ps(diff_w_low[location_index]);
                  __m512 coeff2 = _mm512_sub_ps(one_f, coeff1);
                  __m512 res1 = _mm512_mul_ps(data_tl, coeff2);
                  res1 = _mm512_fmadd_ps(data_tr, coeff1, res1);
                  __m512 res2 = _mm512_mul_ps(data_bl, coeff2);
                  res2 = _mm512_fmadd_ps(data_br, coeff1, res2);
                  coeff1 = _mm512_set1_ps(diff_h_low[location_index]);
                  coeff2 = _mm512_sub_ps(one_f, coeff1);
                  res1 = _mm512_mul_ps(res1, coeff2);
                  res1 = _mm512_fmadd_ps(res2, coeff1, res1);
                  accumulator = _mm512_fmadd_ps(weight, res1, accumulator);

                  location_index++;
              }
              _mm512_storeu_ps(output + output_index, accumulator);
          }
        }

        feature_map_begin += feature_map_height_i32 * feature_map_width_i32 * M * D;
      }
    };

    concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, threadCount, worker_lambda);

    alloc->Free(buffer);
  }
}  // namespace contrib
}  // namespace onnxruntime
