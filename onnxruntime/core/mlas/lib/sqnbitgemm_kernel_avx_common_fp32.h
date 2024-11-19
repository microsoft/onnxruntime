#pragma once
#include "qnbitgemm.h"

template <bool HasZeroPoint>
MLAS_FORCEINLINE
    size_t
    MlasQ4GemmKernelBlkLen16Avx512f(
        const float* A,
        const std::byte* QuantBData,
        const float* QuantBScale,
        const std::byte* QuantBZeroPoint,
        float* C,
        size_t CountM,
        size_t CountN,
        size_t CountK,
        size_t BlockCountK,
        const float* Bias,
        size_t lda,
        size_t ldc
    )
{
    // We process 32 quantized values in a batch.
    // assert(BlkLen % 32 == 0)
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols = 4;
    constexpr size_t BlkLen16 = 16;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    const __m128i lowMask = _mm_set1_epi8(0xF);

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer

    for (size_t m = 0; m < CountM; m++) {
        //*//
        ////const float* BiasPtr = Bias;

        // for each row of A, reset B pointers
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        ////float* SumPtr = CRowPtr;
        //*//

        auto* sum_ptr = C;
        const auto* bias_ptr = Bias;

        int64_t nblk = (int64_t)(CountN)-4;
        while (nblk >= 0) {
            __m512 acc_lo0 = _mm512_setzero_ps();
            __m512 acc_lo1 = _mm512_setzero_ps();
            __m512 acc_lo2 = _mm512_setzero_ps();
            __m512 acc_lo3 = _mm512_setzero_ps();

            //*//
            const std::byte* b_blk_data_ptr = QuantBDataColPtr;
            const float* s = QuantBScaleColPtr;
            //*//

            if constexpr (HasZeroPoint) {
                QuantBZeroPointIdx = 0;
            }

            for (size_t k = 0; k < CountK; k += BlkLen16) {
                size_t kklen = std::min(CountK - k, BlkLen16);

                const float scale_v0 = *(s);
                const float scale_v1 = *(s + StrideQuantBScale * 1);
                const float scale_v2 = *(s + StrideQuantBScale * 2);
                const float scale_v3 = *(s + StrideQuantBScale * 3);

                const __m128i* b0ptr = (const __m128i*)(b_blk_data_ptr);
                const __m128i* b1ptr = (const __m128i*)(b_blk_data_ptr + StrideQuantBData * 1);
                const __m128i* b2ptr = (const __m128i*)(b_blk_data_ptr + StrideQuantBData * 2);
                const __m128i* b3ptr = (const __m128i*)(b_blk_data_ptr + StrideQuantBData * 3);

                // Load A row vector of 16 floats
                uint32_t mask = 0xffff >> (BlkLen16 - kklen);
                __m512 av_lo = _mm512_maskz_loadu_ps(__mmask16(mask), A + k);

                // Load B col vectors of 16 of 4b
                // SubBlkLen = 16: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
                const __m128i bvi4_0 = _mm_loadl_epi64(b0ptr++);
                const __m128i bvi4_1 = _mm_loadl_epi64(b1ptr++);
                const __m128i bvi4_2 = _mm_loadl_epi64(b2ptr++);
                const __m128i bvi4_3 = _mm_loadl_epi64(b3ptr++);

                // expand 4b into byte array
                __m128i lower = _mm_and_si128(bvi4_0, lowMask);
                __m128i upper = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bvi4_0, 4), lowMask), 8);
                __m128i bytes0 = _mm_add_epi8(upper, lower);

                lower = _mm_and_si128(bvi4_1, lowMask);
                upper = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bvi4_1, 4), lowMask), 8);
                __m128i bytes1 = _mm_add_epi8(upper, lower);

                lower = _mm_and_si128(bvi4_2, lowMask);
                upper = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bvi4_2, 4), lowMask), 8);
                __m128i bytes2 = _mm_add_epi8(upper, lower);

                lower = _mm_and_si128(bvi4_3, lowMask);
                upper = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bvi4_3, 4), lowMask), 8);
                __m128i bytes3 = _mm_add_epi8(upper, lower);

                // Subtract zero-point from the integers
                if constexpr (HasZeroPoint) {
                    // Subtract zero-point from the integers
                    bool is_lower = (QuantBZeroPointIdx & 1) == 0;

                    // TODO: void condition on is_lower
                    std::byte zp_packed = QuantBZeroPointColPtr[0 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                    uint8_t zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));

                    bytes0 = _mm_sub_epi8(bytes0, _mm_set1_epi8(zp));

                    zp_packed = QuantBZeroPointColPtr[1 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                    zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
                    bytes1 = _mm_sub_epi8(bytes1, _mm_set1_epi8(zp));

                    zp_packed = QuantBZeroPointColPtr[2 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                    zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
                    bytes2 = _mm_sub_epi8(bytes2, _mm_set1_epi8(zp));

                    zp_packed = QuantBZeroPointColPtr[3 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                    zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
                    bytes3 = _mm_sub_epi8(bytes3, _mm_set1_epi8(zp));
                } else {
                    // Subtract 8 from the integers
                    const __m128i eight = _mm_set1_epi8(8);
                    bytes0 = _mm_sub_epi8(bytes0, eight);
                    bytes1 = _mm_sub_epi8(bytes1, eight);
                    bytes2 = _mm_sub_epi8(bytes2, eight);
                    bytes3 = _mm_sub_epi8(bytes3, eight);
                }

                // Convert to 16-bit int
                const __m256i vx16_0 = _mm256_cvtepi8_epi16(bytes0);
                const __m256i vx16_1 = _mm256_cvtepi8_epi16(bytes1);
                const __m256i vx16_2 = _mm256_cvtepi8_epi16(bytes2);
                const __m256i vx16_3 = _mm256_cvtepi8_epi16(bytes3);

                // Convert to 32-bit int -> float 32
                __m512 bvf_0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_0));
                __m512 bvf_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_1));
                __m512 bvf_2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_2));
                __m512 bvf_3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_3));

                __m512 scale_ps = _mm512_set1_ps(scale_v0);
                bvf_0 = _mm512_mul_ps(bvf_0, scale_ps);
                scale_ps = _mm512_set1_ps(scale_v1);
                bvf_1 = _mm512_mul_ps(bvf_1, scale_ps);
                scale_ps = _mm512_set1_ps(scale_v2);
                bvf_2 = _mm512_mul_ps(bvf_2, scale_ps);
                scale_ps = _mm512_set1_ps(scale_v3);
                bvf_3 = _mm512_mul_ps(bvf_3, scale_ps);

                acc_lo0 = _mm512_fmadd_ps(bvf_0, av_lo, acc_lo0);
                acc_lo1 = _mm512_fmadd_ps(bvf_1, av_lo, acc_lo1);
                acc_lo2 = _mm512_fmadd_ps(bvf_2, av_lo, acc_lo2);
                acc_lo3 = _mm512_fmadd_ps(bvf_3, av_lo, acc_lo3);

                //*//
                b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
                s++;

                if constexpr (HasZeroPoint) {
                    QuantBZeroPointIdx += 1;
                }
                //*//

            }  // k

            __m128 acc_x = FoldAccumulators(acc_lo0, acc_lo1, acc_lo2, acc_lo3);
            if (Bias != nullptr) {
                acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(bias_ptr));
            }
            _mm_storeu_ps(sum_ptr, acc_x);

            // move to next 4 columns
            sum_ptr += 4;
            bias_ptr += 4;
            nblk -= 4;

            //*//
            QuantBDataColPtr += NCols * StrideQuantBData;
            QuantBScaleColPtr += NCols * StrideQuantBScale;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
            }

            ////BiasPtr += BiasPtr != nullptr ? NCols : 0;
            ////SumPtr += NCols;

            ////nblk -= NCols;
            //*//
        }

        // left over columns less than 4 ?
        nblk += 4;
        if (nblk > 0) {
            __m512 acc_lo[4]{};

            //*//
            const std::byte* b_blk_data_ptr = QuantBDataColPtr;
            const float* s = QuantBScaleColPtr;
            //*//

            if constexpr (HasZeroPoint) {
                QuantBZeroPointIdx = 0;
            }

            for (size_t k = 0; k < CountK; k += BlkLen16) {
                size_t klen = std::min(CountK - k, BlkLen16);

                float scale_v[4];
                const __m128i* b_ptr[4];
                for (int64_t nn = 0; nn < nblk; nn++) {
                    //*//
                    scale_v[nn] = *(s + StrideQuantBScale * nn);
                    b_ptr[nn] = (const __m128i*)(b_blk_data_ptr + StrideQuantBData * nn);
                    //*//
                }

                uint32_t mask = 0xffff >> (BlkLen16 - klen);
                __m512 av_lo = _mm512_maskz_loadu_ps(__mmask16(mask), A + k);

                for (int64_t nn = 0; nn < nblk; nn++) {
                    // Load B col vectors of 16 of 4b
                    // SubBlkLen = 16: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
                    const __m128i bvi4_0 = _mm_loadl_epi64(b_ptr[nn]++);

                    // expand 4b into byte array
                    __m128i lower = _mm_and_si128(bvi4_0, lowMask);
                    __m128i upper = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bvi4_0, 4), lowMask), 8);
                    __m128i bytes = _mm_add_epi8(upper, lower);

                    if constexpr (HasZeroPoint) {
                        // Subtract zero-point from the integers
                        bool is_lower = (QuantBZeroPointIdx & 1) == 0;

                        // TODO: void condition on is_lower
                        std::byte zp_packed = QuantBZeroPointColPtr[nn * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                        uint8_t zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
                        bytes = _mm_sub_epi8(bytes, _mm_set1_epi8(zp));
                    } else {
                        // Subtract 8 from the integers
                        const __m128i eight = _mm_set1_epi8(8);
                        bytes = _mm_sub_epi8(bytes, eight);
                    }

                    // Convert to 16-bit int
                    const __m256i vx16 = _mm256_cvtepi8_epi16(bytes);

                    // Convert to 32-bit int -> float 32
                    __m512 bvf = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16));
                    __m512 scale_16_ps = _mm512_set1_ps(scale_v[nn]);
                    bvf = _mm512_mul_ps(bvf, scale_16_ps);

                    acc_lo[nn] = _mm512_fmadd_ps(bvf, av_lo, acc_lo[nn]);
                }

                //*//
                b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
                s++;

                if constexpr (HasZeroPoint) {
                    QuantBZeroPointIdx += 1;
                }
                //*//
            }  // k

            for (int64_t nn = 0; nn < nblk; nn++) {
                sum_ptr[nn] = _mm512_reduce_add_ps(acc_lo[nn]);
                sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
            }
        }

        // Prepare pointers for the next row
        C += ldc;
        A += lda;
    }
    return CountM;
}

template <bool HasZeroPoint, bool IsBlkLen64Layout>
MLAS_FORCEINLINE
    size_t
    MlasQ4GemmKernelBlkLen32PlusAvx512f(
        size_t BlkLen,
        const float* A,
        const std::byte* QuantBData,
        const float* QuantBScale,
        const std::byte* QuantBZeroPoint,
        float* C,
        size_t CountM,
        size_t CountN,
        size_t CountK,
        size_t BlockCountK,
        const float* Bias,
        size_t lda,
        size_t ldc
    )
{
    // We process 32 quantized values in a batch.
    // assert(BlkLen % 32 == 0)
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols = 4;
    constexpr size_t MLAS_QUANT4_BLK_UNIT32 = 32;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    const __m256i lowMask = _mm256_set1_epi8(0xF);

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer

    for (size_t m = 0; m < CountM; m++) {
        //*//
        ////const float* BiasPtr = Bias;

        // for each row of A, reset B pointers
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        ////float* SumPtr = CRowPtr;
        //*//

        auto* sum_ptr = C;
        const auto* bias_ptr = Bias;

        int64_t nblk = (int64_t)(CountN)-4;
        while (nblk >= 0) {
            __m512 acc_lo0 = _mm512_setzero_ps();
            __m512 acc_lo1 = _mm512_setzero_ps();
            __m512 acc_lo2 = _mm512_setzero_ps();
            __m512 acc_lo3 = _mm512_setzero_ps();

            //*//
            const std::byte* b_blk_data_ptr = QuantBDataColPtr;
            const float* s = QuantBScaleColPtr;
            //*//

            if constexpr (HasZeroPoint) {
                QuantBZeroPointIdx = 0;
            }

            for (size_t k = 0; k < CountK; k += BlkLen) {
                size_t ck = std::min(CountK - k, BlkLen);

                const float scale_v0 = *(s);
                const float scale_v1 = *(s + StrideQuantBScale * 1);
                const float scale_v2 = *(s + StrideQuantBScale * 2);
                const float scale_v3 = *(s + StrideQuantBScale * 3);

                const __m128i* b0ptr = (const __m128i*)(b_blk_data_ptr);
                const __m128i* b1ptr = (const __m128i*)(b_blk_data_ptr + StrideQuantBData * 1);
                const __m128i* b2ptr = (const __m128i*)(b_blk_data_ptr + StrideQuantBData * 2);
                const __m128i* b3ptr = (const __m128i*)(b_blk_data_ptr + StrideQuantBData * 3);

                for (size_t kk = 0; kk < ck; kk += MLAS_QUANT4_BLK_UNIT32) {
                    size_t kklen = std::min((size_t)MLAS_QUANT4_BLK_UNIT32, ck - kk);

                    // Load A row vectors
                    uint32_t mask = 0xffffffff >> (MLAS_QUANT4_BLK_UNIT32 - kklen);
                    __m512 av_lo = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);

                    mask = mask >> 16;
                    __m512 av_hi = mask == 0 ? _mm512_setzero_ps()
                                             : _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk + 16);

                    // Load B col vectors
                    __m256i bytes0, bytes1, bytes2, bytes3;
                    if constexpr (IsBlkLen64Layout) {
                        // dst: | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
                        // load 64 weights at once, parse to get v0 - v31 if subblk is even, otherwise get v32 - v63
                        // increment b_data_ptr by 2 * MLAS_QUANT4_BLK_UNIT32 if subblk is odd
                        // so that all v0-63 of the pack are processed.
                        const __m256i bvi4_0 = _mm256_loadu_si256((__m256i const*)(b0ptr));
                        const __m256i bvi4_1 = _mm256_loadu_si256((__m256i const*)(b1ptr));
                        const __m256i bvi4_2 = _mm256_loadu_si256((__m256i const*)(b2ptr));
                        const __m256i bvi4_3 = _mm256_loadu_si256((__m256i const*)(b3ptr));
                        const int count_half_4 =
                            4 * ((kk % (2 * MLAS_QUANT4_BLK_UNIT32)) / MLAS_QUANT4_BLK_UNIT32);
                        bytes0 = _mm256_and_si256(_mm256_srli_epi16(bvi4_0, count_half_4), lowMask);
                        bytes1 = _mm256_and_si256(_mm256_srli_epi16(bvi4_1, count_half_4), lowMask);
                        bytes2 = _mm256_and_si256(_mm256_srli_epi16(bvi4_2, count_half_4), lowMask);
                        bytes3 = _mm256_and_si256(_mm256_srli_epi16(bvi4_3, count_half_4), lowMask);
                        b0ptr += count_half_4 / 2;
                        b1ptr += count_half_4 / 2;
                        b2ptr += count_half_4 / 2;
                        b3ptr += count_half_4 / 2;
                    } else {
                        const __m128i bvi4_0 = _mm_loadu_si128(b0ptr++);
                        const __m128i bvi4_1 = _mm_loadu_si128(b1ptr++);
                        const __m128i bvi4_2 = _mm_loadu_si128(b2ptr++);
                        const __m128i bvi4_3 = _mm_loadu_si128(b3ptr++);

                        // expand 4b into byte array
                        bytes0 = _mm256_set_m128i(_mm_srli_epi16(bvi4_0, 4), bvi4_0);
                        bytes1 = _mm256_set_m128i(_mm_srli_epi16(bvi4_1, 4), bvi4_1);
                        bytes2 = _mm256_set_m128i(_mm_srli_epi16(bvi4_2, 4), bvi4_2);
                        bytes3 = _mm256_set_m128i(_mm_srli_epi16(bvi4_3, 4), bvi4_3);
                        bytes0 = _mm256_and_si256(lowMask, bytes0);
                        bytes1 = _mm256_and_si256(lowMask, bytes1);
                        bytes2 = _mm256_and_si256(lowMask, bytes2);
                        bytes3 = _mm256_and_si256(lowMask, bytes3);
                    }

                    // Subtract zero-point from the integers
                    if constexpr (HasZeroPoint) {
                        // Subtract zero-point from the integers
                        bool is_lower = (QuantBZeroPointIdx & 1) == 0;

                        // TODO: void condition on is_lower
                        std::byte zp_packed = QuantBZeroPointColPtr[0 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                        uint8_t zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));

                        bytes0 = _mm256_sub_epi8(bytes0, _mm256_set1_epi8(zp));

                        zp_packed = QuantBZeroPointColPtr[1 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                        zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
                        bytes1 = _mm256_sub_epi8(bytes1, _mm256_set1_epi8(zp));

                        zp_packed = QuantBZeroPointColPtr[2 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                        zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
                        bytes2 = _mm256_sub_epi8(bytes2, _mm256_set1_epi8(zp));

                        zp_packed = QuantBZeroPointColPtr[3 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                        zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
                        bytes3 = _mm256_sub_epi8(bytes3, _mm256_set1_epi8(zp));
                    } else {
                        // Subtract 8 from the integers
                        const __m256i eight = _mm256_set1_epi8(8);
                        bytes0 = _mm256_sub_epi8(bytes0, eight);
                        bytes1 = _mm256_sub_epi8(bytes1, eight);
                        bytes2 = _mm256_sub_epi8(bytes2, eight);
                        bytes3 = _mm256_sub_epi8(bytes3, eight);
                    }

                    // Convert to 16-bit int
                    const __m256i vx16_lo0 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 0));
                    const __m256i vx16_hi0 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 1));
                    const __m256i vx16_lo1 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 0));
                    const __m256i vx16_hi1 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 1));
                    const __m256i vx16_lo2 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 0));
                    const __m256i vx16_hi2 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 1));
                    const __m256i vx16_lo3 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 0));
                    const __m256i vx16_hi3 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 1));

                    // Convert to 32-bit int -> float 32
                    __m512 bvf_lo0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo0));
                    __m512 bvf_hi0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi0));
                    __m512 bvf_lo1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo1));
                    __m512 bvf_hi1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi1));
                    __m512 bvf_lo2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo2));
                    __m512 bvf_hi2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi2));
                    __m512 bvf_lo3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo3));
                    __m512 bvf_hi3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi3));

                    __m512 scale_ps = _mm512_set1_ps(scale_v0);
                    bvf_lo0 = _mm512_mul_ps(bvf_lo0, scale_ps);
                    bvf_hi0 = _mm512_mul_ps(bvf_hi0, scale_ps);
                    scale_ps = _mm512_set1_ps(scale_v1);
                    bvf_lo1 = _mm512_mul_ps(bvf_lo1, scale_ps);
                    bvf_hi1 = _mm512_mul_ps(bvf_hi1, scale_ps);
                    scale_ps = _mm512_set1_ps(scale_v2);
                    bvf_lo2 = _mm512_mul_ps(bvf_lo2, scale_ps);
                    bvf_hi2 = _mm512_mul_ps(bvf_hi2, scale_ps);
                    scale_ps = _mm512_set1_ps(scale_v3);
                    bvf_lo3 = _mm512_mul_ps(bvf_lo3, scale_ps);
                    bvf_hi3 = _mm512_mul_ps(bvf_hi3, scale_ps);

                    acc_lo0 = _mm512_fmadd_ps(bvf_lo0, av_lo, acc_lo0);
                    acc_lo0 = _mm512_fmadd_ps(bvf_hi0, av_hi, acc_lo0);
                    acc_lo1 = _mm512_fmadd_ps(bvf_lo1, av_lo, acc_lo1);
                    acc_lo1 = _mm512_fmadd_ps(bvf_hi1, av_hi, acc_lo1);
                    acc_lo2 = _mm512_fmadd_ps(bvf_lo2, av_lo, acc_lo2);
                    acc_lo2 = _mm512_fmadd_ps(bvf_hi2, av_hi, acc_lo2);
                    acc_lo3 = _mm512_fmadd_ps(bvf_lo3, av_lo, acc_lo3);
                    acc_lo3 = _mm512_fmadd_ps(bvf_hi3, av_hi, acc_lo3);
                }  // kk

                //*//
                b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
                s++;

                if constexpr (HasZeroPoint) {
                    QuantBZeroPointIdx += 1;
                }
                //*//

            }  // k

            __m128 acc_x = FoldAccumulators(acc_lo0, acc_lo1, acc_lo2, acc_lo3);
            if (Bias != nullptr) {
                acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(bias_ptr));
            }
            _mm_storeu_ps(sum_ptr, acc_x);

            // move to next 4 columns
            sum_ptr += 4;
            bias_ptr += 4;
            nblk -= 4;

            //*//
            QuantBDataColPtr += NCols * StrideQuantBData;
            QuantBScaleColPtr += NCols * StrideQuantBScale;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
            }

            ////BiasPtr += BiasPtr != nullptr ? NCols : 0;
            ////SumPtr += NCols;

            ////nblk -= NCols;
            //*//
        }

        // left over columns less than 4 ?
        nblk += 4;
        if (nblk > 0) {
            __m512 acc_lo[4]{};

            //*//
            const std::byte* b_blk_data_ptr = QuantBDataColPtr;
            const float* s = QuantBScaleColPtr;
            //*//

            if constexpr (HasZeroPoint) {
                QuantBZeroPointIdx = 0;
            }

            for (size_t k = 0; k < CountK; k += BlkLen) {
                size_t ck = std::min(CountK - k, BlkLen);

                float scale_v[4];
                const __m128i* b_ptr[4];
                for (int64_t nn = 0; nn < nblk; nn++) {
                    //*//
                    scale_v[nn] = *(s + StrideQuantBScale * nn);
                    b_ptr[nn] = (const __m128i*)(b_blk_data_ptr + StrideQuantBData * nn);
                    //*//
                }

                for (size_t kk = 0; kk < ck; kk += MLAS_QUANT4_BLK_UNIT32) {
                    size_t kklen = std::min((size_t)MLAS_QUANT4_BLK_UNIT32, ck - kk);

                    uint32_t mask = 0xffffffff >> (MLAS_QUANT4_BLK_UNIT32 - kklen);
                    __m512 av_lo = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);

                    mask = mask >> 16;
                    __m512 av_hi = mask == 0
                                       ? _mm512_setzero_ps()
                                       : _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk + 16);

                    for (int64_t nn = 0; nn < nblk; nn++) {
                        __m256i bytes;
                        if constexpr (IsBlkLen64Layout) {
                            // dst: | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
                            // load 64 weights at once, parse to get v0 - v31 if subblk is even, otherwise get v32 - v63
                            // increment b_data_ptr by 2 * MLAS_QUANT4_BLK_UNIT32 if subblk is odd
                            // so that all v0-63 of the pack are processed.
                            const __m256i bvi4 = _mm256_loadu_si256((__m256i const*)(b_ptr[nn]));
                            const int count_half_4 =
                                4 * ((kk % (2 * MLAS_QUANT4_BLK_UNIT32)) / MLAS_QUANT4_BLK_UNIT32);
                            bytes = _mm256_and_si256(_mm256_srli_epi16(bvi4, count_half_4), lowMask);
                            b_ptr[nn] += count_half_4 / 2;
                        } else {
                            const __m128i bvi4 = _mm_loadu_si128(b_ptr[nn]++);
                            bytes = _mm256_set_m128i(_mm_srli_epi16(bvi4, 4), bvi4);
                            bytes = _mm256_and_si256(lowMask, bytes);
                        }
                        if constexpr (HasZeroPoint) {
                            // Subtract zero-point from the integers
                            bool is_lower = (QuantBZeroPointIdx & 1) == 0;

                            // TODO: void condition on is_lower
                            std::byte zp_packed = QuantBZeroPointColPtr[nn * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                            uint8_t zp = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
                            bytes = _mm256_sub_epi8(bytes, _mm256_set1_epi8(zp));
                        } else {
                            // Subtract 8 from the integers
                            const __m256i eight = _mm256_set1_epi8(8);
                            bytes = _mm256_sub_epi8(bytes, eight);
                        }

                        // Convert to 16-bit int
                        const __m256i vx16_lo =
                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 0));
                        const __m256i vx16_hi =
                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 1));

                        // Convert to 32-bit int -> float 32
                        __m512 bvf_lo = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo));
                        __m512 bvf_hi = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi));
                        __m512 scale_16_ps = _mm512_set1_ps(scale_v[nn]);
                        bvf_lo = _mm512_mul_ps(bvf_lo, scale_16_ps);
                        bvf_hi = _mm512_mul_ps(bvf_hi, scale_16_ps);

                        acc_lo[nn] = _mm512_fmadd_ps(bvf_lo, av_lo, acc_lo[nn]);
                        acc_lo[nn] = _mm512_fmadd_ps(bvf_hi, av_hi, acc_lo[nn]);
                    }
                }  // kk

                //*//
                b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
                s++;

                if constexpr (HasZeroPoint) {
                    QuantBZeroPointIdx += 1;
                }
                //*//
            }  // k

            for (int64_t nn = 0; nn < nblk; nn++) {
                sum_ptr[nn] = _mm512_reduce_add_ps(acc_lo[nn]);
                sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
            }
        }

        // Prepare pointers for the next row
        C += ldc;
        A += lda;
    }
    return CountM;
}
