/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    half gemm.cpp

Abstract:

    This module implements the half precision (fp16) matrix/matrix multiply
    operation (QGEMM).

--*/

#include "mlasi.h"
#include "mlas_float16.h"

#include "halfgemm.h"

#include <exception>

bool MLASCALL
MlasFp16AccelerationSupported()
{
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
    return MLAS_CPUIDINFO::GetCPUIDInfo().HasFp16VectorAcceleration();
#else
    return false;
#endif
}


void
MLASCALL
MlasHalfGemmBatch(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_HALF_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool
    )
{
    const MLAS_HALFGEMM_DISPATCH* dispatch = MlasHalfGemmGetDispatch();
    MLAS_HALFGEMM_OPERATION* operation = dispatch->Operation;

    if (ThreadPool == nullptr) {
        for (size_t gemm_i = 0; gemm_i < BatchN; gemm_i++) {
            auto Data = &DataParams[gemm_i];
            operation(N, K, Data, 0, M, 0, N);
        }
        return;
    }

    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K) * double(BatchN);

    ptrdiff_t TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    ptrdiff_t ThreadsPerGemm = TargetThreadCount / BatchN;
    if (ThreadsPerGemm < 1) {
        ThreadsPerGemm = 1;
    }

    const size_t StrideM = dispatch->StrideM;

    size_t nc = N;
    if ((size_t)MlasGetMaximumThreadCount(ThreadPool) > BatchN) {
        // more than one thread per GEMM

        const size_t BlockedM = MlasDivRoundup(M, StrideM);
        const size_t max_nc = MlasDivRoundup(N * BlockedM, ThreadsPerGemm);
        if (max_nc < nc) {
            nc = std::min(nc, MlasDivRoundup(nc, max_nc * MLAS_QGEMM_STRIDEN_THREAD_ALIGN) *
                                  MLAS_QGEMM_STRIDEN_THREAD_ALIGN);
        }
    }
    const size_t StrideN = nc;

    const size_t ThreadCountM = MlasDivRoundup(M, StrideM);
    const size_t ThreadCountN = MlasDivRoundup(N, StrideN);
    ThreadsPerGemm = ThreadCountM * ThreadCountN;

    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN, [&](ptrdiff_t tid) {
        const auto gemm_i = tid / ThreadsPerGemm;
        const auto blk_i = tid % ThreadsPerGemm;
        auto Data = &DataParams[gemm_i];

        const ptrdiff_t ThreadIdN = blk_i / ThreadCountM;
        const ptrdiff_t ThreadIdM = blk_i % ThreadCountM;

        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = std::min(M - RangeStartM, (size_t)StrideM);

        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = std::min(N - RangeStartN, (size_t)StrideN);

        operation(N, K, Data, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
    });
}


size_t
MLASCALL
MlasHalfGemmPackBSize(
    size_t N,
    size_t K,
    bool float2half
    )
{
    const auto* dispatch = MlasHalfGemmGetDispatch();
    const auto padding = dispatch->BufOverRead;
    const auto PackedK = dispatch->PackededK;
    if (!float2half && dispatch->CopyPackBRoutine == nullptr) {
        // No packing routine provided
        return 0;
    }
    const size_t AlignedK = (K + PackedK - 1) & ~(PackedK - 1);
    const size_t BytesRequired = N * AlignedK * FP16_SIZE + padding;
    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();
    const size_t AlignedBytesRequired =
        (BytesRequired + BufferAlignment - 1) & ~(BufferAlignment - 1);
    return AlignedBytesRequired;
}

void
MLASCALL
MlasHalfGemmPackB(
    size_t N,
    size_t K,
    const MLAS_FP16* B,
    size_t ldb,
    void* PackedB
    )
{
    const auto* dispatch = MlasHalfGemmGetDispatch();
    dispatch->CopyPackBRoutine((_mlas_fp16_*)PackedB, (const _mlas_fp16_*)B, ldb, N, K);
}

void
MLASCALL
MlasHalfGemmConvertPackB(
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB
    )
{
    const auto* dispatch = MlasHalfGemmGetDispatch();
    dispatch->ConvertPackBRoutine((_mlas_fp16_*)PackedB, B, ldb, N, K);
}


//
//  Post Processor Implementations
//

MLAS_FORCEINLINE
void
CvtHalf2Float(
    float* dest,
    const _mlas_fp16_* src,
    size_t len
)
{
#ifdef MLAS_TARGET_ARM64
    while (len >= 4) {
        const auto* srcPtr = reinterpret_cast<const float16x4_t*>(src);
        auto* dstPtr = reinterpret_cast<float32x4_t*>(dest);
        *dstPtr = vcvt_f32_f16(*srcPtr);
        src += 4;
        dest += 4;
        len -= 4;
    }

    if (0 == len) {
        return;
    }

    float16x4_t buf;
    std::memcpy(&buf, src, len * sizeof(_mlas_fp16_));
    float32x4_t res = vcvt_f32_f16(buf);

    if ((len & 2) != 0) {
        auto wide = vreinterpretq_f64_f32(res);
        vst1q_lane_f64((float64_t*)dest, wide, 0);
        res = vreinterpretq_f32_f64(vdupq_laneq_f64(wide, 1));
        dest += 2;
    }
    if ((len & 1) != 0) {
        vst1q_lane_f32(dest, res, 0);
    }
#else
    for (size_t i = 0; i < len; i++) {
        *dest++ = MLAS_Half2Float(*src++);
    }
#endif  // MLAS_TARGET_ARM64
}

void
MLAS_HALF_GEMM_2FLOAT_PROCESSOR::Process(
    MLAS_FP16* C,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN,
    size_t ldc
    ) const
{
    float* Output = Output_;
    const auto* CRow = reinterpret_cast<const _mlas_fp16_*>(C);
    CRow += StartM * ldc + StartN;
    Output += StartM * RowStride_ + StartN;

    while (CountM-- > 0) {
        CvtHalf2Float(Output, CRow, CountN);
        MlasActivation(&Activation_, Output, nullptr, 1, CountN, ldc);
        CRow += ldc;
        Output += RowStride_;
    }
}


//
// Dummy C++ implementation that runs very slowly
//

struct MLAS_HALF_GEMM_KERNEL_DEFAULT {

    static constexpr bool PackNeeded = false;
    static constexpr size_t KernelMaxM = 128; // max # rows the vectorized kernel can process
    static constexpr size_t PackedK = 1;

    static constexpr MLAS_HALF_GEMM_STRIDES Strides{8, 16, 32};
};

template<>
MLAS_FORCEINLINE
void
MlasHalfGemmConvertPackA<MLAS_HALF_GEMM_KERNEL_DEFAULT>(
    _mlas_fp16_* D,
    const float* A,
    size_t lda,
    size_t CountM,
    size_t CountK
)
{
    for (size_t m = 0; m < CountM; m++) {
        for (size_t k = 0; k < CountK; k++) {
            *D++ = MLAS_Float2Half(*(A + m * lda + k));
        }
    }
}

template<>
MLAS_FORCEINLINE
void
MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_DEFAULT>(
    _mlas_fp16_* D,
    const float* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
)
{
    for (size_t k = 0; k < CountK; k++) {
        for (size_t n = 0; n < CountN; n++) {
            *D++ = MLAS_Float2Half(*(B + k * ldb + n));
        }
    }
}


template<>
MLAS_FORCEINLINE
void
MlasHalfGemmKernel<MLAS_HALF_GEMM_KERNEL_DEFAULT>(
    size_t CountM,
    size_t CountN,
    size_t CountK,
    _mlas_fp16_* C,
    size_t ldc,
    const _mlas_fp16_* Bias,
    const _mlas_fp16_* A,
    size_t lda,
    const _mlas_fp16_* B,
    size_t ldb,
    const bool ZeroMode)
{
    for (size_t m = 0; m < CountM; m++) {
        for (size_t n = 0; n < CountN; n++) {
            const auto* a = A + (m * lda);
            const auto* b = B + n;
            auto* c = C + (m * ldc) + n;

            float sum = Bias == nullptr ? 0.0f : MLAS_Half2Float(Bias[n]);
            if (!ZeroMode) {
                sum += MLAS_Half2Float(*c);
            }

            for (size_t k = 0; k < CountK; k++) {
                auto down = MLAS_Float2Half(MLAS_Half2Float(*a) * MLAS_Half2Float(*b) + sum);
                sum = MLAS_Half2Float(down);
                b += ldb;
                a += 1;
            }

            *c = MLAS_Float2Half(sum);
        }
    }
}

bool
MLASCALL
MlasHGemmSupported(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB
) {
    auto* dispatch = GetMlasPlatform().HGemmDispatch;
    if (TransA == CblasNoTrans && TransB == CblasTrans) {
        return dispatch &&
        dispatch->HGemmKernel_TransposedB &&
        dispatch->HPackBKernel_TransposedB &&
        dispatch->HGemmKernel_PackedB;
    } else if (TransA == CblasNoTrans && TransB == CblasNoTrans) {
        return dispatch &&
        dispatch->HGemmKernel_B &&
        dispatch->HPackBKernel_B &&
        dispatch->HGemmKernel_PackedB;
    }

    return false;
}

void
HGemmOperation(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t K, // full K slice
    const MLAS_HGEMM_DATA_PARAMS* DataParams,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
) {
    const size_t lda = DataParams->lda;
    const size_t ldb = DataParams->ldb;
    const size_t ldc = DataParams->ldc;
    const _mlas_fp16_ alpha = DataParams->alpha;
    const _mlas_fp16_ beta = DataParams->beta;
    auto* dispatch = GetMlasPlatform().HGemmDispatch;
    constexpr size_t StrideM = 2;
    const auto beta_add = MLAS_FP16(1.0f);
    constexpr size_t buffer_size = MLAS_HGEMM_STRIDEN * MLAS_HGEMM_STRIDEK;

    if (TransA == CblasNoTrans && TransB == CblasTrans) {
        const auto* A = DataParams->A + RangeStartM * lda;
        const auto* B = DataParams->B + RangeStartN * ldb;
        auto* C = DataParams->C + RangeStartM * ldc + RangeStartN;

        if (RangeCountM <= StrideM) {
            if (!dispatch || !dispatch->HGemmKernel_TransposedB) {
                MLAS_THROW_EX(std::runtime_error, "hgemm does not have A x Transposed(B) kernels");
            }
            // When M is small, B is visited once. The overhead of Pack(B') exceeds the benefits
            // from A x Pack(B'). Therefore directly calculate A x B'.
            // Without PackB, to utilize memory locality, iterate full K.
            constexpr size_t StrideN = MLAS_HGEMM_STRIDEN_THREAD_ALIGN;
            for (size_t n = 0, countN; n < RangeCountN; n += countN) {
                countN = std::min(StrideN, RangeCountN - n);
                dispatch->HGemmKernel_TransposedB(A, B, C, RangeCountM, countN, K, lda, ldb, ldc, alpha, beta);
                B += countN * ldb;
                C += countN;
            }
        } else {
            if (!dispatch || !dispatch->HPackBKernel_TransposedB || !dispatch->HGemmKernel_PackedB) {
                MLAS_THROW_EX(std::runtime_error, "hgemm does not have A x Transposed(B) kernels");
            }
            // 16N is the smallest pack unit.
            // TODO(fajin): optimize alpha == 1
            MLAS_DECLSPEC_ALIGN(MLAS_FP16 PackedB[buffer_size], MLAS_HGEMM_STRIDEN_THREAD_ALIGN * sizeof(_mlas_fp16_));
            size_t StrideN = MLAS_HGEMM_STRIDEN;
            size_t StrideK = MLAS_HGEMM_STRIDEK;
            if (RangeCountN >= K) {
                while (StrideK / 2 >= K) {
                    StrideN *= 2;
                    StrideK /= 2;
                }

            } else {
                while (StrideN > MLAS_HGEMM_STRIDEN_THREAD_ALIGN && StrideN / 2 >= RangeCountN) {
                    StrideK *= 2;
                    StrideN /= 2;
                }
            }

            for (size_t n = 0, countN; n < RangeCountN; n += countN) {
                countN = std::min(StrideN, RangeCountN - n);
                const MLAS_FP16* a = A;
                const MLAS_FP16* b = B;
                MLAS_FP16* c = C;
                for (size_t k = 0, countK; k < K; k += countK) {
                    countK = std::min(StrideK, K - k);
                    dispatch->HPackBKernel_TransposedB(b, PackedB, countN, countK, ldb);
                    const MLAS_FP16* aa = a;
                    MLAS_FP16* cc = c;
                    for (size_t m = 0, countM; m < RangeCountM; m += countM) {
                        countM = std::min(StrideM, RangeCountM - m);
                        // First K iteration, beta is applied to the whole C. In rest K iterations, use add mode.
                        dispatch->HGemmKernel_PackedB(
                            aa, PackedB, cc, countM, countN, countK, lda, ldc, alpha, k == 0 ? beta : beta_add.val);
                        aa += countM * lda;
                        cc += countM * ldc;
                    }
                    a += countK;
                    b += countK;
                }
                B += countN * ldb;
                C += countN;
            }
        }
    } else if (TransA == CblasNoTrans && TransB == CblasNoTrans) {
        const auto* A = DataParams->A + RangeStartM * lda;
        const auto* B = DataParams->B + RangeStartN;
        auto* C = DataParams->C + RangeStartM * ldc + RangeStartN;

        if (RangeCountM <= StrideM) {
            if (!dispatch || !dispatch->HGemmKernel_B) {
                MLAS_THROW_EX(std::runtime_error, "hgemm does not have A x B kernels");
            }

            // When M is small, B is visited once. The overhead of Pack(B) exceeds the benefits
            // from A x Pack(B). Therefore directly calculate A x B.
            // When beta is 0 or 1, iterate full N and cache accumulators in C.
            // When beta is not 0 or 1, iterate full K, accumulat in register, max 8 accumulators.
            // TODO(fajin): merge beta cases with alpha == 1
            dispatch->HGemmKernel_B(A, B, C, RangeCountM, RangeCountN, K, lda, ldb, ldc, alpha, beta);
        } else {
            if (!dispatch || !dispatch->HPackBKernel_B || !dispatch->HGemmKernel_PackedB) {
                MLAS_THROW_EX(std::runtime_error, "hgemm does not have A x B kernels");
            }
            // TODO(fajin): optimize blocking for large K small N
            //  - pack along N
            //  - loop K in outer loop
            //  - optimize alpha == 1 case
            MLAS_DECLSPEC_ALIGN(MLAS_FP16 PackedB[buffer_size], MLAS_HGEMM_STRIDEN_THREAD_ALIGN * sizeof(_mlas_fp16_));
            size_t StrideN = MLAS_HGEMM_STRIDEN;
            size_t StrideK = MLAS_HGEMM_STRIDEK;
            if (RangeCountN >= K) {
                while (StrideK / 2 >= K) {
                    StrideN *= 2;
                    StrideK /= 2;
                }
            } else {
                while (StrideN > MLAS_HGEMM_STRIDEN_THREAD_ALIGN && StrideN / 2 >= RangeCountN) {
                    StrideK *= 2;
                    StrideN /= 2;
                }
            }

            for (size_t n = 0, countN; n < RangeCountN; n += countN) {
                countN = std::min(StrideN, RangeCountN - n);
                const MLAS_FP16* a = A;
                const MLAS_FP16* b = B;
                MLAS_FP16* c = C;
                for (size_t k = 0, countK; k < K; k += countK) {
                    countK = std::min(StrideK, K - k);
                    dispatch->HPackBKernel_B(b, PackedB, countN, countK, ldb);
                    const MLAS_FP16* aa = a;
                    MLAS_FP16* cc = c;
                    for (size_t m = 0, countM; m < RangeCountM; m += countM) {
                        countM = std::min(StrideM, RangeCountM - m);
                        // First K iteration, beta is applied to the whole C. In rest K iterations, use add mode.
                        dispatch->HGemmKernel_PackedB(
                            aa, PackedB, cc, countM, countN, countK, lda, ldc, alpha, k == 0 ? beta : beta_add.val);
                        aa += countM * lda;
                        cc += countM * ldc;
                    }
                    a += countK;
                    b += countK * ldb;
                }
                B += countN;
                C += countN;
            }
        }
    } else {
        MLAS_THROW_EX(std::runtime_error, "hgemm currently only support A x Transpoe(B) or A x B");
    }
}

void
MLASCALL
MlasGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_HGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
) {
    if (!ThreadPool) {
        for (size_t gemm_i = 0; gemm_i < BatchSize; gemm_i++) {
            HGemmOperation(TransA, TransB, K, &Data[gemm_i], 0, M, 0, N);
        }
        return;
    }

    const double Complexity = double(M) * double(N) * double(K) * double(BatchSize);
    ptrdiff_t TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_HGEMM_THREAD_COMPLEXITY)) + 1;
    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    // Segment the operation across multiple threads.

    ptrdiff_t ThreadsPerGemm = TargetThreadCount / BatchSize;
    if (ThreadsPerGemm < 1) {
        ThreadsPerGemm = 1;
    }

    constexpr size_t StrideM = 128;

    size_t nc = N;
    if (ThreadsPerGemm > 1) {
        // more than one thread per GEMM

        const size_t BlockedM = MlasDivRoundup(M, StrideM);
        const size_t max_nc = MlasDivRoundup(N * BlockedM, ThreadsPerGemm);
        if (max_nc < nc) {
            nc = std::min(
                nc, MlasDivRoundup(max_nc, MLAS_HGEMM_STRIDEN_THREAD_ALIGN) * MLAS_HGEMM_STRIDEN_THREAD_ALIGN);
        }
    }
    const size_t StrideN = nc;

    const size_t ThreadCountM = MlasDivRoundup(M, StrideM);
    const size_t ThreadCountN = MlasDivRoundup(N, StrideN);
    ThreadsPerGemm = ThreadCountM * ThreadCountN;

    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * static_cast<ptrdiff_t>(BatchSize), [&](ptrdiff_t tid) {
        const auto gemm_i = tid / ThreadsPerGemm;
        const auto blk_i = tid % ThreadsPerGemm;

        const ptrdiff_t ThreadIdN = blk_i / ThreadCountM;
        const ptrdiff_t ThreadIdM = blk_i % ThreadCountM;

        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = std::min(M - RangeStartM, (size_t)StrideM);

        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = std::min(N - RangeStartN, (size_t)StrideN);

        HGemmOperation(TransA, TransB, K, &Data[gemm_i], RangeStartM, RangeCountM, RangeStartN, RangeCountN);
    });
}

const MLAS_HALFGEMM_DISPATCH MlasHalfGemmDispatchDefault = {
    MlasHalfGemmOperation<MLAS_HALF_GEMM_KERNEL_DEFAULT>,
    nullptr,
    MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_DEFAULT>,
    MLAS_HALF_GEMM_KERNEL_DEFAULT::PackedK,
    MLAS_HALF_GEMM_KERNEL_DEFAULT::KernelMaxM,
    0
};
