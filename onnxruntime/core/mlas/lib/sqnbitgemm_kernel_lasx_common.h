/*++
    Abstract:

        Lasx/Lsx tool function, Auxiliary functions for inference required by
        4-bit/8-bit quantization models.
--*/
#pragma once
#include "qnbitgemm.h"
#include "core/common/safeint.h"
#include <memory>

template<typename T, size_t RequiredAlignment = 0>
struct MlasAlignedAllocator {
    using value_type = T;

    MlasAlignedAllocator() = default;

    template<typename U, size_t A>
    MlasAlignedAllocator(const MlasAlignedAllocator<U, A>&) {}

    T* allocate(size_t n) {
        // If RequiredAlignment > 0, use the required value
        // Otherwise, use the value of MlasGetPreferredBufferAlignment()
        size_t alignment = RequiredAlignment > 0 ?
                          RequiredAlignment :
                          MlasGetPreferredBufferAlignment();

        size_t size = n * sizeof(T);
        if (size % alignment != 0)  // check the size
            size = ((size + alignment - 1) / alignment) * alignment;
        #if defined(_MSC_VER)
            void* ptr = _aligned_malloc(size, alignment);
        #else
            void* ptr = aligned_alloc(alignment, size);
        #endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t) {
        #if defined(_MSC_VER)
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
    }

    template<typename U>
    struct rebind {
        using other = MlasAlignedAllocator<U, RequiredAlignment>;
    };
};

static MLAS_FORCEINLINE __m256
__lasx_xvzero()
{
    return (__m256)__lasx_xvldi(0);
}

static size_t
GetContinueLayoutOffsetSubBlk(size_t N, const size_t n, const size_t SubOrBlkCountK, const size_t k_sub_or_blk)
{
    size_t T = n / 4, t = n % 4;
    bool te = T == N / 4;
    SafeInt<size_t> scale_dst_offset = SafeInt<size_t>(T) * 4 * SubOrBlkCountK;
    if (te) {
        scale_dst_offset +=  SafeInt<size_t>(t) * SubOrBlkCountK + k_sub_or_blk;
    } else {
        scale_dst_offset +=  SafeInt<size_t>(k_sub_or_blk) * 4 + t;
    }
    return scale_dst_offset.Value();
}

static size_t
GetContinueLayoutOffsetBlkInSubBlk(size_t N, const size_t n, const size_t BlockCountK, const size_t k_blk, const int blks_per_sub)
{
    size_t T = n / 4, t = n % 4, k_subblk = k_blk / blks_per_sub, b = k_blk % blks_per_sub;
    bool te = T == N / 4, be = k_subblk == BlockCountK / blks_per_sub;
    SafeInt<size_t> scale_dst_offset =  SafeInt<size_t>(T) * 4 * BlockCountK;
    if (te) {
        scale_dst_offset +=  SafeInt<size_t>(t) * BlockCountK + k_blk;
    } else {
        scale_dst_offset +=  SafeInt<size_t>(k_subblk) * blks_per_sub * 4;
        if (be) {
            scale_dst_offset +=  SafeInt<size_t>(b) * 4 + t;
        } else {
            scale_dst_offset +=  SafeInt<size_t>(t) * blks_per_sub + b;
        }
    }
    return scale_dst_offset.Value();
}

static void
ComputePackBlkSum_Lasx(
    size_t BlkLen,
    size_t SubBlkLen,
    size_t N,
    float* QuantBScaleBegin,
    const std::byte* QuantBZPBegin,
    float* BlockSumBegin,
    MLAS_THREADPOOL* ThreadPool,
    const size_t BlockCountK
)
{
    MlasTrySimpleParallel(ThreadPool, N * BlockCountK, [&](ptrdiff_t tid) {
        const size_t n = tid / BlockCountK;
        const size_t k_blk = tid % BlockCountK;

        const SafeInt<size_t> src_blk_offset =  SafeInt<size_t>(n) * BlockCountK + k_blk;
        float QuantBScale = QuantBScaleBegin[src_blk_offset.Value()];
        uint8_t zp = 8;

        if (QuantBZPBegin) {
            size_t ZPCountK = MlasDivRoundup(BlockCountK, 2);
            SafeInt<size_t> src_zp_offset =  SafeInt<size_t>(ZPCountK) * n + k_blk / 2;
            bool low_zp = k_blk % 2 == 0;
            const std::byte* QuantBZP = QuantBZPBegin + src_zp_offset.Value();
            const std::byte low_mask{0X0f};
            zp = (uint8_t)(low_zp ? ((*QuantBZP) & low_mask) : ((*QuantBZP) >> 4));
        }

        float result = -QuantBScale * zp;

        const SafeInt<size_t> dst_offset = ( SafeInt<size_t>(n / 16) * BlockCountK + k_blk) * 16 + n % 16;
        BlockSumBegin[dst_offset.Value()] = result;

        if (BlkLen == 16) {
        } else if (BlkLen >= SubBlkLen) {
            const size_t scale_dst_offset = GetContinueLayoutOffsetSubBlk(N, n, BlockCountK, k_blk);
            QuantBScaleBegin[scale_dst_offset] = QuantBScale;
        } else {
            int blks_per_sub = (int)(SubBlkLen / BlkLen);
            size_t scale_dst_offset = GetContinueLayoutOffsetBlkInSubBlk(N, n, BlockCountK, k_blk, blks_per_sub);
            QuantBScaleBegin[scale_dst_offset] = QuantBScale;
        }
    });
}

static void
PackQuantB(
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool,
    const size_t N,
    const size_t BlockCountK,
    const size_t BlkLen,
    const size_t SubBlkLen
)
{
    constexpr size_t BlkBitWidth = 4;
    const size_t BlkBytePairCount = BlkLen / 4;
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);

    const size_t SubBlkDataSize = SubBlkLen / 2;
    const size_t SubBlkBytePairCount = SubBlkLen / 4;
    const size_t SubBlkCountK = MlasDivRoundup(BlockCountK * BlkLen, SubBlkLen);
    const size_t Iterations = N * SubBlkCountK;  // one iteration per sub block

    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            const size_t n = tid / SubBlkCountK;
            const size_t k_subblk = tid % SubBlkCountK;

            const SafeInt<size_t> src_data_offset =  SafeInt<size_t>(n) * BlockCountK * BlkDataSize + k_subblk * SubBlkDataSize;
            const std::byte* QuantBData = QuantBDataBegin + src_data_offset.Value();

            size_t PackBytePairCount = SubBlkBytePairCount;
            size_t PackDataSize = SubBlkDataSize;

            auto pack_subblk = [](
                                   const std::byte* QuantBData, std::byte* PackedQuantBData,
                                   size_t pack_byte_pair_count, size_t pack_data_size
                               ) {
            for (size_t byte_pair_idx = 0; byte_pair_idx < pack_byte_pair_count; ++byte_pair_idx) {
                const std::byte src0 = QuantBData[byte_pair_idx];
                const std::byte src1 = QuantBData[byte_pair_idx + pack_data_size / 2];

                std::byte& dst0 = PackedQuantBData[2 * byte_pair_idx];
                std::byte& dst1 = PackedQuantBData[2 * byte_pair_idx + 1];

                dst0 = (src0 & std::byte{0x0f}) | ((src1 & std::byte{0x0f}) << 4);
                dst1 = (src0 >> 4) | ((src1 >> 4) << 4);
            } };

            if (SubBlkLen > BlkLen && k_subblk == SubBlkCountK - 1 &&
                SubBlkLen * SubBlkCountK > BlkLen * BlockCountK) {
                // this is the last subblk of the column. check if it extends out of the
                // BlockCountK. If it does, we shall pack per blocks so that can compute
                // on each block instead of each subblk.
                PackBytePairCount = BlkBytePairCount;
                PackDataSize = BlkDataSize;
                const size_t k_blks_remaining = BlockCountK - (SubBlkCountK - 1) * SubBlkLen / BlkLen;
                for (size_t k = 0; k < k_blks_remaining; k++) {
                    const SafeInt<size_t> k_blk =  SafeInt<size_t>(k_subblk) * SubBlkLen / BlkLen + k;
                    if (BlkLen == 16) {
                        // not to do the compute order layout yet
                        std::byte* PackedQuantBData = PackedQuantBDataBegin + src_data_offset;
                        pack_subblk(QuantBData + k * BlkLen / 2, PackedQuantBData + k * BlkLen / 2, PackBytePairCount, PackDataSize);
                    } else if (BlkLen >= SubBlkLen) {
                        // shall not reach here with avx2
                        assert(SubBlkLen == 128);
                    } else {
                        int blks_per_sub = (int)(SubBlkLen / BlkLen);
                        const size_t dst_data_offset = GetContinueLayoutOffsetBlkInSubBlk(N, n, BlockCountK, k_blk.Value(), blks_per_sub);
                        std::byte* PackedQuantBData = PackedQuantBDataBegin + dst_data_offset * BlkLen / 2;
                        pack_subblk(QuantBData + k * BlkLen / 2, PackedQuantBData, PackBytePairCount, PackDataSize);
                    }
                }
            } else {
                if (BlkLen == 16) {
                    // not to do the compute order layout yet
                    std::byte* PackedQuantBData = PackedQuantBDataBegin + src_data_offset;
                    pack_subblk(QuantBData, PackedQuantBData, PackBytePairCount, PackDataSize);
                } else if (BlkLen >= SubBlkLen) {
                    const size_t dst_data_offset = GetContinueLayoutOffsetSubBlk(N, n, SubBlkCountK, k_subblk);
                    std::byte* PackedQuantBData = PackedQuantBDataBegin + dst_data_offset * SubBlkDataSize;
                    pack_subblk(QuantBData, PackedQuantBData, PackBytePairCount, PackDataSize);
                } else {
                    int blks_per_sub = (int)(SubBlkLen / BlkLen);
                    const SafeInt<size_t> k_blk =  SafeInt<size_t>(k_subblk) * blks_per_sub;
                    const size_t dst_data_offset = GetContinueLayoutOffsetBlkInSubBlk(N, n, BlockCountK, k_blk.Value(), blks_per_sub);
                    std::byte* PackedQuantBData = PackedQuantBDataBegin + dst_data_offset * BlkLen / 2;
                    pack_subblk(QuantBData, PackedQuantBData, PackBytePairCount, PackDataSize);
                }
            }
        }
    );
}

template <typename IterationFn, size_t... Indices>
MLAS_FORCEINLINE void
UnrolledLoopIterations(IterationFn&& f, std::index_sequence<Indices...> /* indices */)
{
    (f(Indices), ...);
}

template <size_t N, typename IterationFn>
MLAS_FORCEINLINE void
UnrolledLoop(IterationFn&& f)
{
    UnrolledLoopIterations(std::forward<IterationFn>(f), std::make_index_sequence<N>());
}

static MLAS_FORCEINLINE __m128
FoldAccumulators_Lasx(const __m256& acc0, const __m256& acc1, const __m256& acc2, const __m256& acc3)
{
    /*
      acc0 = [A0, A1, A2, A3, A4, A5, A6, A7]
      acc1 = [B0, B1, B2, B3, B4, B5, B6, B7]
    */

    __m256 tmpAB_lo = (__m256)__lasx_xvpermi_d(__lasx_xvpermi_w(acc1, acc0, 0x44), 0xD8);  // a1,a2,a5,a6,b1,b2,b5,b6
    __m256 tmpAB_hi = (__m256)__lasx_xvpermi_d(__lasx_xvpermi_w(acc1, acc0, 0xEE), 0xD8);  // a3,a4,a7,a8,b3,b4,b7,b8
    __m256 tmpCD_lo = (__m256)__lasx_xvpermi_d(__lasx_xvpermi_w(acc3, acc2, 0x44), 0xD8);  // c1,c2,c5,c6,d1,d2,d5,d6
    __m256 tmpCD_hi = (__m256)__lasx_xvpermi_d(__lasx_xvpermi_w(acc3, acc2, 0xEE), 0xD8);  // c3,c4,c7,c8,d3,d4,d7,d8

    __m256 tmpABCD_lo1 = (__m256)__lasx_xvpermi_w(tmpCD_lo, tmpAB_lo, 0x44);  // a1,a2,c1,c2,b1,b2,d1,d2
    __m256 tmpABCD_lo2 = (__m256)__lasx_xvpermi_w(tmpCD_hi, tmpAB_hi, 0x44);  // a3,a4,c3,c4,b3,b4,d3,d4
    __m256 tmpABCD_hi1 = (__m256)__lasx_xvpermi_w(tmpCD_lo, tmpAB_lo, 0xEE);  // a5,a6,c5,c6,b5,b6,d5,d6
    __m256 tmpABCD_hi2 = (__m256)__lasx_xvpermi_w(tmpCD_hi, tmpAB_hi, 0xEE);  // a7,a8,c7,c8,b7,b8,d7,d8

    __m256 sumABCD = __lasx_xvfadd_s(__lasx_xvfadd_s(tmpABCD_lo1, tmpABCD_lo2), __lasx_xvfadd_s(tmpABCD_hi1, tmpABCD_hi2));

    __m256 sum0 = (__m256)__lasx_xvpermi_w(sumABCD, sumABCD, 0xB1);
    sumABCD = (__m256)__lasx_xvpermi_d(__lasx_xvfadd_s(sumABCD, sum0), 0xD8);

    sumABCD = (__m256)__lasx_xvpermi_d(__lasx_xvpermi_w(sumABCD, sumABCD, 0x88), 0xD8);

    alignas(32) float tmp[8];
    __lasx_xvst(sumABCD, (void*)&tmp, 0);
    __m128 result = (__m128)__lsx_vld((void*)&tmp, 0);
    return result;
}

__m256
permutevar_ps_lasx(__m256 vec, __m256i idx_mask)
{
    __m256i veci = (__m256i)vec;
    __m256i shuffled = __lasx_xvshuf_w(veci, veci, idx_mask);
    return (__m256)shuffled;
}

static void
Q8PackQuantB(
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool,
    const size_t N,
    const size_t BlockCountK,
    const size_t BlkLen,
    const size_t SubBlkLen
)
{
    constexpr size_t BlkBitWidth = 8;
    const size_t StrideN = BlockCountK * BlkLen;
    const size_t BlkSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t SubBlkSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, SubBlkLen);
    const size_t SubBlkCountK = MlasDivRoundup(StrideN, SubBlkLen);
    const size_t RemainderBlockCountK = BlockCountK % (SubBlkLen > BlkLen ? SubBlkLen / BlkLen : 1);
    const size_t Iterations = N * SubBlkCountK;  // one iteration per sub block

    // SubBlkLen rows x 4 columns pack together, then remainder BlkLen x 4 columns if SubBlkLen > BlkLen.
    // remainder columns keep the original order.
    // SubBlkLen >= 16 and is multiple of 16

    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            const size_t c = tid / SubBlkCountK;
            const size_t c_4 = c & (~3), c_res = c & 3;
            const size_t r_subblk = tid % SubBlkCountK;

            const SafeInt<size_t> data_offset =  SafeInt<size_t>(c) * StrideN + r_subblk * SubBlkLen;
            const std::byte* src = QuantBDataBegin + data_offset.Value();

            if (c_4 + 4 <= N) {                                              // full 4 cols
                if (RemainderBlockCountK && r_subblk == SubBlkCountK - 1) {  // remainder blocks
                    const SafeInt<size_t> subblk_data_offset =  SafeInt<size_t>(c_4) * StrideN + r_subblk * SubBlkSize * 4 + c_res * BlkSize;
                    std::byte* dest =
                        PackedQuantBDataBegin + subblk_data_offset.Value();
                    for (size_t i = 0; i < RemainderBlockCountK; i++) {
                        std::copy(src, src + BlkSize, dest);
                        src += BlkSize;
                        dest += BlkSize * 4;
                    }
                } else {  // full subblock
                    const SafeInt<size_t> subblk_data_offset =  SafeInt<size_t>(c_4) * StrideN + r_subblk * SubBlkSize * 4 + c_res * SubBlkSize;
                    std::byte* dest =
                        PackedQuantBDataBegin + subblk_data_offset.Value();
                    std::copy(src, src + SubBlkSize, dest);
                }
            } else {  // remainder cols
                const SafeInt<size_t> remain_data_offset =  SafeInt<size_t>(c) * StrideN + r_subblk * SubBlkSize;
                std::byte* dest =
                    PackedQuantBDataBegin + remain_data_offset.Value();
                std::copy(src, src + std::min(SubBlkSize, StrideN - r_subblk * SubBlkSize), dest);
            }
        }
    );
}

static void
Q8ComputePackBlkSum(
    size_t BlkLen,
    size_t SubBlkLen,
    size_t N,
    float* QuantBScaleBegin,
    const std::byte* QuantBZPBegin,
    float* BlockSumBegin,
    MLAS_THREADPOOL* ThreadPool,
    const size_t BlockCountK
)
{
    SafeInt<size_t> size =  SafeInt<size_t>(N) * BlockCountK;
    std::vector<float, MlasAlignedAllocator<float, 32>> QuantBScaleBeginCopy(size.Value());
    std::copy(QuantBScaleBegin, QuantBScaleBegin + N * BlockCountK, QuantBScaleBeginCopy.begin());

    MlasTrySimpleParallel(ThreadPool, N * BlockCountK, [&](ptrdiff_t tid) {
        const size_t n = tid / BlockCountK;
        const size_t n_4 = n & (~3), n_res = n & 3;
        const size_t k_blk = tid % BlockCountK;

        const SafeInt<size_t> src_blk_offset =  SafeInt<size_t>(n) * BlockCountK + k_blk;
        const float& QuantBScale = QuantBScaleBeginCopy[src_blk_offset.Value()];
        uint8_t zp = 128;
        if (QuantBZPBegin) {
            const std::byte* QuantBZP = QuantBZPBegin + src_blk_offset.Value();
            zp = (uint8_t)(*QuantBZP);
        }

        const SafeInt<size_t> dst_offset = ( SafeInt<size_t>(n / 16) * BlockCountK + k_blk) * 16 + n % 16;
        *(BlockSumBegin + dst_offset.Value()) = -QuantBScale * zp;

        if (n_4 + 4 > N) {
            SafeInt<size_t> ptr_offset =  SafeInt<size_t>(n) * BlockCountK + k_blk;
            *(QuantBScaleBegin + ptr_offset.Value()) = QuantBScale;
        } else if (BlkLen >= SubBlkLen) {
            SafeInt<size_t> ptr_offset =  SafeInt<size_t>(n_4) * BlockCountK + k_blk * 4 + n_res;
            *(QuantBScaleBegin + ptr_offset.Value()) = QuantBScale;
        } else {
            size_t blks_per_sub = SubBlkLen / BlkLen;
            size_t remainder_blk = BlockCountK % blks_per_sub;
            size_t sub_blk_count_k = MlasDivRoundup(BlockCountK, blks_per_sub);
            size_t k_subblk = k_blk / blks_per_sub;
            size_t k_blk_res = k_blk % blks_per_sub;
            SafeInt<size_t> dest_offset;

            if (remainder_blk && k_subblk == sub_blk_count_k - 1) {  // remainder blocks
                dest_offset =  SafeInt<size_t>(n_4) * BlockCountK + k_blk * 4 + n_res;
            } else {  // full subblock
                dest_offset =  SafeInt<size_t>(n_4) * BlockCountK + k_subblk * blks_per_sub * 4 + n_res * blks_per_sub + k_blk_res;
            }

            *(QuantBScaleBegin + dest_offset.Value()) = QuantBScale;
        }
    });
}

static void
Q8PackQuantBDataAndBlkSum_lasx(
    size_t N,
    size_t BlockCountK,
    size_t BlkLen,
    size_t SubBlkLen,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool HasZeroPoint,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 8>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool
)
{
    if (QuantBDataBegin) {
        Q8PackQuantB(QuantBDataBegin, PackedQuantB.PackedQuantBData, ThreadPool, N, BlockCountK, BlkLen, SubBlkLen);
    }

    if (QuantBScaleBegin) {
        std::copy(QuantBScaleBegin, QuantBScaleBegin + N * BlockCountK, PackedQuantB.PackedQuantBScale);
    }

    if ((QuantBScaleBegin && !HasZeroPoint) || QuantBZPBegin) {
        Q8ComputePackBlkSum(BlkLen, SubBlkLen, N, PackedQuantB.PackedQuantBScale, QuantBZPBegin, PackedQuantB.QuantBBlkSum, ThreadPool, BlockCountK);
    }
}

static MLAS_FORCEINLINE __m128i
convert_2_ps_to_epi8_lasx(__m256 v0, __m256 v1)
{
    // fp32->int32
    __m256i v0_8_epi32 = __lasx_xvftint_w_s(__lasx_xvfrint_s(v0));
    __m256i v1_8_epi32 = __lasx_xvftint_w_s(__lasx_xvfrint_s(v1));

    alignas(32) int val_0_15_i32[16] = {0};
    alignas(32) int8_t val_0_15_i8[16] = {0};

    __lasx_xvst(v0_8_epi32, (void*)&val_0_15_i32, 0);
    __lasx_xvst(v1_8_epi32, (void*)&val_0_15_i32, 32);

    UnrolledLoop<16>([&](size_t i) {
        if (val_0_15_i32[i] > 127)
            val_0_15_i8[i] = 127;
        else if (val_0_15_i32[i] < -128)
            val_0_15_i8[i] = -128;
        else
            val_0_15_i8[i] = static_cast<int8_t>(val_0_15_i32[i]);
    });

    __m128i result = __lsx_vld((void*)&val_0_15_i8, 0);
    return result;
}

static inline __m256i
lasx_maddubs_epi16_sat(__m256i a, __m256i b)
{
    // a: bytes interpreted as unsigned
    // b: bytes interpreted as signed
    __m256i zero_h = __lasx_xvldi(0);  // 256-bit zeros

    __m256i even_prod16 = __lasx_xvmaddwev_h_bu_b(zero_h, a, b);
    __m256i odd_prod16 = __lasx_xvmaddwod_h_bu_b(zero_h, a, b);

    __m256i sum16_sat = __lasx_xvsadd_h(even_prod16, odd_prod16);

    return sum16_sat;  // 16-bit signed saturated results (16 lanes)
}

static inline __m256i
lasx_madd_epi16(__m256i a, __m256i b)
{
    __m256i zero = __lasx_xvldi(0);
    __m256i even_acc = __lasx_xvmaddwev_w_h(zero, a, b);
    __m256i result = __lasx_xvmaddwod_w_h(even_acc, a, b);

    return result;  // 32-bit signed sums, matches _mm256_madd_epi16 semantics (no saturation)
}

static inline __m256i
lasx_hadd_epi32(__m256i a, __m256i b)
{
    __m256i a_swapped = __lasx_xvshuf4i_w(a, 0xB1);  // 0xB1 = binary 10110001
    __m256i b_swapped = __lasx_xvshuf4i_w(b, 0xB1);

    __m256i a_sum = __lasx_xvadd_w(a, a_swapped);
    __m256i b_sum = __lasx_xvadd_w(b, b_swapped);

    __m256i a_even = __lasx_xvpermi_w(a_sum, a_sum, 0x88);
    __m256i b_even = __lasx_xvpermi_w(b_sum, b_sum, 0x88);

    __m256i result = __lasx_xvpermi_q(a_even, b_even, 0x20);

    return result;
}

static inline __m256i
lasx_cvtepu8_epi16_emul_from_m128(const __m128i v128)
{
    alignas(32) int8_t num[32] = {0};
    __lsx_vst(v128, (void*)&num, 0);
    __m256i result = __lasx_xvld((void*)&num, 0);
    result = __lasx_xvexth_hu_bu(__lasx_xvpermi_d(result, 0x72));
    return result;
}

static MLAS_FORCEINLINE float
hsum_float_8_lasx(__m256 v)
{
    v = __lasx_xvfadd_s(v, (__m256)__lasx_xvpermi_d(v, 0xB1));
    v = __lasx_xvfadd_s(v, (__m256)__lasx_xvpermi_d(v, 0x4E));
    alignas(32) float num[8] = {0.0f};
    __lasx_xvst(v, (void*)num, 0);

    return num[0] + num[1];
}
