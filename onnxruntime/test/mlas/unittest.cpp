/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    unittest.cpp

Abstract:

    This module implements unit tests of the MLAS library.

--*/

#include <stdio.h>
#include <memory.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <mlas.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#if !defined(MLAS_NO_ONNXRUNTIME_THREADPOOL)
#include "core/platform/threadpool.h"
#endif

#include "core/common/make_unique.h"

#if !defined(_countof)
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif

MLAS_THREADPOOL* threadpool = nullptr;

template<typename T>
class MatrixGuardBuffer
{
public:
    MatrixGuardBuffer()
    {
        _BaseBuffer = nullptr;
        _BaseBufferSize = 0;
        _ElementsAllocated = 0;
    }

    ~MatrixGuardBuffer(void)
    {
        ReleaseBuffer();
    }

    T* GetBuffer(size_t Elements, bool ZeroFill = false)
    {
        //
        // Check if the internal buffer needs to be reallocated.
        //

        if (Elements > _ElementsAllocated) {

            ReleaseBuffer();

            //
            // Reserve a virtual address range for the allocation plus an unmapped
            // guard region.
            //

            constexpr size_t BufferAlignment = 64 * 1024;
            constexpr size_t GuardPadding = 256 * 1024;

            size_t BytesToAllocate = ((Elements * sizeof(T)) + BufferAlignment - 1) & ~(BufferAlignment - 1);

            _BaseBufferSize = BytesToAllocate + GuardPadding;

#if defined(_WIN32)
            _BaseBuffer = VirtualAlloc(NULL, _BaseBufferSize, MEM_RESERVE, PAGE_NOACCESS);
#else
            _BaseBuffer = mmap(0, _BaseBufferSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

            if (_BaseBuffer == nullptr) {
                abort();
            }

            //
            // Commit the number of bytes for the allocation leaving the upper
            // guard region as unmapped.
            //

#if defined(_WIN32)
            if (VirtualAlloc(_BaseBuffer, BytesToAllocate, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
                ORT_THROW_EX(std::bad_alloc);
            }
#else
            if (mprotect(_BaseBuffer, BytesToAllocate, PROT_READ | PROT_WRITE) != 0) {
                abort();
            }
#endif

            _ElementsAllocated = BytesToAllocate / sizeof(T);
            _GuardAddress = (T*)((unsigned char*)_BaseBuffer + BytesToAllocate);
        }

        //
        //
        //

        T* GuardAddress = _GuardAddress;
        T* buffer = GuardAddress - Elements;

        if (ZeroFill) {

            std::fill_n(buffer, Elements, T(0));

        } else {

            const int MinimumFillValue = -23;
            const int MaximumFillValue = 23;

            int FillValue = MinimumFillValue;
            T* FillAddress = buffer;

            while (FillAddress < GuardAddress) {

                *FillAddress++ = (T)FillValue;

                FillValue++;

                if (FillValue > MaximumFillValue) {
                    FillValue = MinimumFillValue;
                }
            }
        }

        return buffer;
    }

    void ReleaseBuffer(void)
    {
        if (_BaseBuffer != nullptr) {

#if defined(_WIN32)
            VirtualFree(_BaseBuffer, 0, MEM_RELEASE);
#else
            munmap(_BaseBuffer, _BaseBufferSize);
#endif

            _BaseBuffer = nullptr;
            _BaseBufferSize = 0;
        }

        _ElementsAllocated = 0;
    }

private:
    size_t _ElementsAllocated;
    void* _BaseBuffer;
    size_t _BaseBufferSize;
    T* _GuardAddress;
};

class MlasTestBase
{
public:
    virtual
    ~MlasTestBase(
        void
        )
    {
    }

    //
    // Contains tests that run quickly as part of a checkin integration to
    // sanity check that the functionality is working.
    //

    virtual
    void
    ExecuteShort(
        void
        )
    {
    }

    //
    // Contains tests that can run slowly to more exhaustively test that
    // functionality is working across a broader range of parameters.
    //

    virtual
    void
    ExecuteLong(
        void
        )
    {
    }
};

template<typename T, bool Packed>
class FgemmPackedContext;

template<typename T>
class FgemmPackedContext<T, false>
{
public:
    void
    TestGemm(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t M,
        size_t N,
        size_t K,
        float alpha,
        const T* A,
        size_t lda,
        const T* B,
        size_t ldb,
        float beta,
        T* C,
        size_t ldc
        )
    {
        MlasGemm(TransA, TransB, M, N, K, T(alpha), A, lda, B, ldb, T(beta), C, ldc, threadpool);
    }
};

template<typename T>
class FgemmPackedContext<T, true>
{
public:
    void
    TestGemm(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t M,
        size_t N,
        size_t K,
        float alpha,
        const T* A,
        size_t lda,
        const T* B,
        size_t ldb,
        float beta,
        T* C,
        size_t ldc
        )
    {
        size_t PackedBSize = MlasGemmPackBSize(N, K);
        void* PackedB = BufferBPacked.GetBuffer(PackedBSize, true);
        MlasGemmPackB(TransB, N, K, B, ldb, PackedB);
        MlasGemm(TransA, M, N, K, T(alpha), A, lda, PackedB, T(beta), C, ldc, threadpool);
    }

private:
    MatrixGuardBuffer<uint8_t> BufferBPacked;
};

template<typename T, bool Packed>
class MlasFgemmTest : public MlasTestBase
{
private:
    void
    Test(
        size_t M,
        size_t N,
        size_t K,
        float alpha,
        float beta
        )
    {
        //
        // Skip the test if the B buffer cannot be packed.
        //

        if (Packed && (N == 0 || K == 0)) {
            return;
        }

        const T* A = BufferA.GetBuffer(K * M);
        const T* B = BufferB.GetBuffer(N * K);
        T* C = BufferC.GetBuffer(N * M);
        T* CReference = BufferCReference.GetBuffer(N * M);

        Test(CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, CReference, N);
        Test(CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, CReference, N);
        Test(CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, CReference, N);
        Test(CblasTrans, CblasTrans, M, N, K, alpha, A, M, B, K, beta, C, CReference, N);
    }

    void
    Test(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t M,
        size_t N,
        size_t K,
        float alpha,
        const T* A,
        size_t lda,
        const T* B,
        size_t ldb,
        float beta,
        T* C,
        T* CReference,
        size_t ldc
        )
    {
        std::fill_n(C, M * N, -0.5f);
        std::fill_n(CReference, M * N, -0.5f);

        PackedContext.TestGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        ReferenceGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, CReference, ldc);

        for (size_t f = 0; f < M * N; f++) {
            // Sensitive to comparing positive/negative zero.
            if (C[f] != CReference[f]) {
                printf("mismatch TransA=%d, TransB=%d, M=%zd, N=%zd, K=%zd, alpha=%f, beta=%f  %f %f!\n", TransA, TransB, M, N, K, alpha, beta, float(C[f]), float(CReference[f]));
                break;
            }
        }
    }

    void
    ReferenceGemm(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t M,
        size_t N,
        size_t K,
        float alpha,
        const T* A,
        size_t lda,
        const T* B,
        size_t ldb,
        float beta,
        T* C,
        size_t ldc
        )
    {
        if (TransA == CblasNoTrans) {

            if (TransB == CblasNoTrans) {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const T* a = A + (m * lda);
                        const T* b = B + n;
                        T* c = C + (m * ldc) + n;
                        T sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            sum += (*b * *a);
                            b += ldb;
                            a += 1;
                        }

                        *c = (*c * beta) + (sum * alpha);
                    }
                }

            } else {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const T* a = A + (m * lda);
                        const T* b = B + (n * ldb);
                        T* c = C + (m * ldc) + n;
                        T sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            sum += (*b * *a);
                            b += 1;
                            a += 1;
                        }

                        *c = (*c * beta) + (sum * alpha);
                    }
                }
            }

        } else {

            if (TransB == CblasNoTrans) {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const T* a = A + m;
                        const T* b = B + n;
                        T* c = C + (m * ldc) + n;
                        T sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            sum += (*b * *a);
                            b += ldb;
                            a += lda;
                        }

                        *c = (*c * beta) + (sum * alpha);
                    }
                }

            } else {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const T* a = A + m;
                        const T* b = B + (n * ldb);
                        T* c = C + (m * ldc) + n;
                        T sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            sum += (*b * *a);
                            b += 1;
                            a += lda;
                        }

                        *c = (*c * beta) + (sum * alpha);
                    }
                }
            }
        }
    }

    MatrixGuardBuffer<T> BufferA;
    MatrixGuardBuffer<T> BufferB;
    MatrixGuardBuffer<T> BufferC;
    MatrixGuardBuffer<T> BufferCReference;
    FgemmPackedContext<T, Packed> PackedContext;

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (size_t b = 0; b < 16; b++) {
            Test(b, b, b, 1.0f, 0.0f);
        }
        for (size_t b = 16; b <= 256; b <<= 1) {
            Test(b, b, b, 1.0f, 0.0f);
        }
        for (size_t b = 256; b < 320; b += 32) {
            Test(b, b, b, 1.0f, 0.0f);
        }

        Test(128, 3072, 768, 1.0f, 0.0f);
        Test(128, 768, 3072, 1.0f, 0.0f);
    }

    void
    ExecuteLong(
        void
        ) override
    {
        static const float multipliers[] = { 0.0f, -0.0f, 0.25f, -0.5f, 1.0f, -1.0f };

        for (size_t N = 1; N < 128; N++) {
            for (size_t K = 1; K < 128; K++) {
                for (size_t a = 0; a < _countof(multipliers); a++) {
                    for (size_t b = 0; b < _countof(multipliers); b++) {
                        Test(1, N, K, multipliers[a], multipliers[b]);
                        Test(N, 1, K, multipliers[a], multipliers[b]);
                    }
                }
            }
        }

        for (size_t a = 0; a < _countof(multipliers); a++) {
            float alpha = multipliers[a];

            for (size_t b = 0; b < _countof(multipliers); b++) {
                float beta = multipliers[b];

                for (size_t M = 16; M < 160; M += 32) {
                    for (size_t N = 16; N < 160; N += 32) {

                        static const size_t ks[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 32, 48, 64, 118, 119, 120, 121, 122, 160, 240, 320 };
                        for (size_t k = 0; k < _countof(ks); k++) {
                            size_t K = ks[k];

                            Test(M, N, K, alpha, beta);
                            Test(M + 1, N, K, alpha, beta);
                            Test(M, N + 1, K, alpha, beta);
                            Test(M + 1, N + 1, K, alpha, beta);
                            Test(M + 3, N + 2, K, alpha, beta);
                            Test(M + 4, N, K, alpha, beta);
                            Test(M, N + 4, K, alpha, beta);
                            Test(M + 4, N + 4, K, alpha, beta);
                            Test(M + 3, N + 7, K, alpha, beta);
                            Test(M + 8, N, K, alpha, beta);
                            Test(M, N + 8, K, alpha, beta);
                            Test(M + 12, N + 12, K, alpha, beta);
                            Test(M + 13, N, K, alpha, beta);
                            Test(M, N + 15, K, alpha, beta);
                            Test(M + 15, N + 15, K, alpha, beta);
                        }
                    }
                    printf("a %zd/%zd b %zd/%zd M %zd\n", a, _countof(multipliers), b, _countof(multipliers), M);
                }
            }
        }

        for (size_t M = 0; M < 160; M++) {
            for (size_t N = 0; N < 160; N++) {
                for (size_t K = 0; K < 160; K++) {
                    Test(M, N, K, 1.0f, 0.0f);
                }
            }
            printf("M %zd\n", M);
        }

        for (size_t M = 160; M < 320; M += 24) {
            for (size_t N = 112; N < 320; N += 24) {
                for (size_t K = 0; K < 16; K++) {
                    Test(M, N, K, 1.0f, 0.0f);
                }
                for (size_t K = 16; K < 160; K += 32) {
                    Test(M, N, K, 1.0f, 0.0f);
                }
            }
            printf("M %zd\n", M);
        }
    }
};

template<bool Packed>
class MlasQgemmU8X8U8X8TestBase : public MlasTestBase
{
private:
    void*
    PackB(
        size_t N,
        size_t K,
        const uint8_t* B,
        size_t ldb,
        bool BIsSigned
        )
    {
        size_t PackedBSize = MlasGemmPackBSize(N, K, BIsSigned);
        void* PackedB = BufferBPacked.GetBuffer(PackedBSize);
        MlasGemmPackB(N, K, B, ldb, BIsSigned, PackedB);
        return PackedB;
    }

protected:
    void
    TestGemm(
        size_t M,
        size_t N,
        size_t K,
        const uint8_t* A,
        size_t lda,
        uint8_t offa,
        const uint8_t* B,
        size_t ldb,
        uint8_t offb,
        bool BIsSigned,
        int32_t* C,
        size_t ldc
        )
    {
        MLAS_GEMM_U8X8_PARAMETERS GemmParameters;

        GemmParameters.M = M;
        GemmParameters.N = N;
        GemmParameters.K = K;
        GemmParameters.A = A;
        GemmParameters.lda = lda;
        GemmParameters.ZeroPointA = offa;
        GemmParameters.ZeroPointB = &offb;
        GemmParameters.BIsSigned = BIsSigned;
        GemmParameters.C = C;
        GemmParameters.ldc = ldc;

        if (Packed) {
            GemmParameters.B = PackB(N, K, B, ldb, BIsSigned);
            GemmParameters.BIsPacked = true;
        } else {
            GemmParameters.B = B;
            GemmParameters.ldb = ldb;
        }

        MlasGemm(&GemmParameters, threadpool);
    }

    void
    TestGemm(
        size_t M,
        size_t N,
        size_t K,
        const uint8_t* A,
        size_t lda,
        uint8_t offa,
        const uint8_t* B,
        size_t ldb,
        const uint8_t* offb,
        bool BIsSigned,
        int32_t* C,
        size_t ldc
        )
    {
        MLAS_GEMM_U8X8_PARAMETERS GemmParameters;

        GemmParameters.M = M;
        GemmParameters.N = N;
        GemmParameters.K = K;
        GemmParameters.A = A;
        GemmParameters.lda = lda;
        GemmParameters.ZeroPointA = offa;
        GemmParameters.ZeroPointB = offb;
        GemmParameters.BIsSigned = BIsSigned;
        GemmParameters.PerColumnZeroPoints = true;
        GemmParameters.C = C;
        GemmParameters.ldc = ldc;

        if (Packed) {
            GemmParameters.B = PackB(N, K, B, ldb, BIsSigned);
            GemmParameters.BIsPacked = true;
        } else {
            GemmParameters.B = B;
            GemmParameters.ldb = ldb;
        }

        MlasGemm(&GemmParameters, threadpool);
    }

    void
    TestGemm(
        size_t M,
        size_t N,
        size_t K,
        const uint8_t* A,
        size_t lda,
        uint8_t offa,
        const uint8_t* B,
        size_t ldb,
        uint8_t offb,
        bool BIsSigned,
        float* C,
        size_t ldc,
        float CScale,
        const float* Bias
        )
    {
        MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR ScaleBiasProcessor(C, ldc, &CScale, Bias);

        MLAS_GEMM_U8X8_PARAMETERS GemmParameters;

        GemmParameters.M = M;
        GemmParameters.N = N;
        GemmParameters.K = K;
        GemmParameters.A = A;
        GemmParameters.lda = lda;
        GemmParameters.ZeroPointA = offa;
        GemmParameters.ZeroPointB = &offb;
        GemmParameters.BIsSigned = BIsSigned;
        GemmParameters.C = reinterpret_cast<int32_t*>(C);
        GemmParameters.ldc = ldc;
        GemmParameters.OutputProcessor = &ScaleBiasProcessor;

        if (Packed) {
            GemmParameters.B = PackB(N, K, B, ldb, BIsSigned);
            GemmParameters.BIsPacked = true;
        } else {
            GemmParameters.B = B;
            GemmParameters.ldb = ldb;
        }

        MlasGemm(&GemmParameters, threadpool);
    }

private:
    MatrixGuardBuffer<uint8_t> BufferBPacked;
};

template<typename xint8_t, typename OutputType, bool Packed>
class MlasQgemmU8X8Test;

template<typename xint8_t, bool Packed>
class MlasQgemmU8X8Test<xint8_t, int32_t, Packed> : public MlasQgemmU8X8U8X8TestBase<Packed>
{
private:
    void
    Test(
        size_t M,
        size_t N,
        size_t K,
        uint8_t offa,
        uint8_t offb
        )
    {
        const uint8_t* A = BufferA.GetBuffer(K * M);
        const uint8_t* B = BufferB.GetBuffer(N * K);
        int32_t* C = BufferC.GetBuffer(N * M);
        int32_t* CReference = BufferCReference.GetBuffer(N * M);

        Test(M, N, K, A, K, offa, B, N, offb, C, CReference, N);
    }

    void
    Test(
        size_t M,
        size_t N,
        size_t K,
        uint8_t offa
        )
    {
        const uint8_t* A = BufferA.GetBuffer(K * M);
        const uint8_t* B = BufferB.GetBuffer(N * K);
        const uint8_t* ZeroPointB = BufferZeroPointB.GetBuffer(N);
        int32_t* C = BufferC.GetBuffer(N * M);
        int32_t* CReference = BufferCReference.GetBuffer(N * M);

        Test(M, N, K, A, K, offa, B, N, ZeroPointB, C, CReference, N);
    }

    void
    Test(
        size_t M,
        size_t N,
        size_t K,
        const uint8_t* A,
        size_t lda,
        uint8_t offa,
        const uint8_t* B,
        size_t ldb,
        uint8_t offb,
        int32_t* C,
        int32_t* CReference,
        size_t ldc
        )
    {
        std::fill_n(C, M * N, -1);
        std::fill_n(CReference, M * N, -1);

        this->TestGemm(M, N, K, A, lda, offa, B, ldb, offb, BIsSigned, C, ldc);
        ReferenceQgemm(M, N, K, A, lda, offa, (const xint8_t*)B, ldb, (xint8_t)offb, CReference, ldc);

        for (size_t f = 0; f < M * N; f++) {
            if (C[f] != CReference[f]) {
                printf("mismatch M=%zd, N=%zd, K=%zd, offa=%d, offb=%d!\n", M, N, K, int(offa), int(offb));
                break;
            }
        }
    }

    void
    Test(
        size_t M,
        size_t N,
        size_t K,
        const uint8_t* A,
        size_t lda,
        uint8_t offa,
        const uint8_t* B,
        size_t ldb,
        const uint8_t *offb,
        int32_t* C,
        int32_t* CReference,
        size_t ldc
        )
    {
        std::fill_n(C, M * N, -1);
        std::fill_n(CReference, M * N, -1);

        this->TestGemm(M, N, K, A, lda, offa, B, ldb, offb, BIsSigned, C, ldc);
        ReferenceQgemm(M, N, K, A, lda, offa, (const xint8_t*)B, ldb, (const xint8_t*)offb, CReference, ldc);

        for (size_t f = 0; f < M * N; f++) {
            if (C[f] != CReference[f]) {
                printf("mismatch M=%zd, N=%zd, K=%zd, offa=%d!\n", M, N, K, int(offa));
                break;
            }
        }
    }

    void
    ReferenceQgemm(
        size_t M,
        size_t N,
        size_t K,
        const uint8_t* A,
        size_t lda,
        uint8_t offa,
        const xint8_t* B,
        size_t ldb,
        xint8_t offb,
        int32_t* C,
        size_t ldc
        )
    {
        for (size_t m = 0; m < M; m++) {

            for (size_t n = 0; n < N; n++) {

                const uint8_t* a = A + (m * lda);
                const xint8_t* b = B + n;
                int32_t* c = C + (m * ldc) + n;
                int32_t sum = 0;

                for (size_t k = 0; k < K; k++) {
                    sum += ((int32_t(*b) - offb) * (int32_t(*a) - offa));
                    b += ldb;
                    a += 1;
                }

                *c = sum;
            }
        }
    }

    void
    ReferenceQgemm(
        size_t M,
        size_t N,
        size_t K,
        const uint8_t* A,
        size_t lda,
        uint8_t offa,
        const xint8_t* B,
        size_t ldb,
        const xint8_t* offb,
        int32_t* C,
        size_t ldc
        )
    {
        for (size_t m = 0; m < M; m++) {

            for (size_t n = 0; n < N; n++) {

                const uint8_t* a = A + (m * lda);
                const xint8_t* b = B + n;
                int32_t* c = C + (m * ldc) + n;
                int32_t sum = 0;

                for (size_t k = 0; k < K; k++) {
                    sum += ((int32_t(*b) - offb[n]) * (int32_t(*a) - offa));
                    b += ldb;
                    a += 1;
                }

                *c = sum;
            }
        }
    }

    MatrixGuardBuffer<uint8_t> BufferA;
    MatrixGuardBuffer<uint8_t> BufferB;
    MatrixGuardBuffer<uint8_t> BufferZeroPointB;
    MatrixGuardBuffer<int32_t> BufferC;
    MatrixGuardBuffer<int32_t> BufferCReference;
    const bool BIsSigned = std::is_signed<xint8_t>::value;

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (size_t b = 1; b < 16; b++) {
            Test(b, b, b, 14, 211);
            Test(b, b, b, 21);
        }
        for (size_t b = 1; b < 16; b++) {
            Test(b, b, b, 14, 211);
            Test(b, b, b, 17);
        }
        for (size_t b = 16; b <= 256; b <<= 1) {
            Test(b, b, b, 34, 1);
            Test(b, b, b, 1);
        }
        for (size_t b = 256; b < 320; b += 32) {
            Test(b, b, b, 85, 173);
        }
        for (size_t b = 1; b < 96; b++) {
            Test(1, b, 32, 0, 0);
            Test(1, 32, b, 0, 0);
            Test(1, b, b, 0, 0);
        }
        Test(43, 500, 401, 183, 223);
        Test(1023, 1023, 1023, 5, 8);
        Test(1023, 1023, 1023, 7);
    }

    void
    ExecuteLong(
        void
        ) override
    {
        static const uint8_t zero_points[] = { 0, 18, 75, 128, 157, 231, 255 };

        for (size_t a = 0; a < _countof(zero_points); a++) {
            uint8_t offa = zero_points[a];

            for (size_t b = 0; b < _countof(zero_points); b++) {
                uint8_t offb = zero_points[b];

                for (size_t M = 16; M < 160; M += 32) {
                    for (size_t N = 16; N < 160; N += 32) {

                        static const size_t ks[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 20, 32, 48, 64, 118, 119, 120, 121, 122, 160, 240, 320 };
                        for (size_t k = 0; k < _countof(ks); k++) {
                            size_t K = ks[k];

                            Test(M, N, K, offa, offb);
                            Test(M + 1, N, K, offa, offb);
                            Test(M, N + 1, K, offa, offb);
                            Test(M + 1, N + 1, K, offa, offb);
                            Test(M + 3, N + 2, K, offa, offb);
                            Test(M + 4, N, K, offa, offb);
                            Test(M, N + 4, K, offa, offb);
                            Test(M + 4, N + 4, K, offa, offb);
                            Test(M + 3, N + 7, K, offa, offb);
                            Test(M + 8, N, K, offa, offb);
                            Test(M, N + 8, K, offa, offb);
                            Test(M + 12, N + 12, K, offa, offb);
                            Test(M + 13, N, K, offa, offb);
                            Test(M, N + 15, K, offa, offb);
                            Test(M + 15, N + 15, K, offa, offb);
                        }
                    }
                    printf("a %zd/%zd b %zd/%zd M %zd\n", a, _countof(zero_points), b, _countof(zero_points), M);
                }
            }
        }

        for (size_t M = 1; M < 160; M++) {
            for (size_t N = 1; N < 160; N++) {
                for (size_t K = 1; K < 160; K++) {
                    Test(M, N, K, 18, 24);
                }
            }
            printf("M %zd\n", M);
        }

        for (size_t M = 160; M < 320; M += 24) {
            for (size_t N = 112; N < 320; N += 24) {
                for (size_t K = 1; K < 16; K++) {
                    Test(M, N, K, 1, 3);
                }
                for (size_t K = 16; K < 160; K += 32) {
                    Test(M, N, K, 5, 7);
                }
            }
            printf("M %zd\n", M);
        }
    }
};

template<typename xint8_t, bool Packed>
class MlasQgemmU8X8Test<xint8_t, float, Packed> : public MlasQgemmU8X8U8X8TestBase<Packed>
{
private:
    void
    Test(
        size_t M,
        size_t N,
        size_t K,
        uint8_t offa,
        uint8_t offb
        )
    {
        const uint8_t* A = BufferA.GetBuffer(K * M);
        const uint8_t* B = BufferB.GetBuffer(N * K);
        float* C = BufferC.GetBuffer(N * M);
        float* CReference = BufferCReference.GetBuffer(N * M);
        const float* Bias = BufferBias.GetBuffer(N);

        const float AScale = 0.5f;
        float* AFloat = BufferAFloat.GetBuffer(K * M);
        DequantizeLinear(A, AFloat, K * M, AScale, offa);

        const float BScale = 0.25f;
        float* BFloat = BufferBFloat.GetBuffer(N * K);
        DequantizeLinear((xint8_t*)B, BFloat, N * K, BScale, xint8_t(offb));

        const float CScale = AScale * BScale;

        Test(M, N, K, A, AFloat, K, offa, B, BFloat, N, offb, C, CReference, N, CScale, nullptr);
        Test(M, N, K, A, AFloat, K, offa, B, BFloat, N, offb, C, CReference, N, CScale, Bias);
    }

    void
    Test(
        size_t M,
        size_t N,
        size_t K,
        const uint8_t* A,
        const float* AFloat,
        size_t lda,
        uint8_t offa,
        const uint8_t* B,
        const float* BFloat,
        size_t ldb,
        uint8_t offb,
        float* C,
        float* CReference,
        size_t ldc,
        float CScale,
        const float* Bias
        )
    {
        MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, AFloat, lda, BFloat, ldb, 0.0f, CReference, ldc, threadpool);

        if (Bias != nullptr) {
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    CReference[m * ldc + n] += Bias[n];
                }
            }
        }

        this->TestGemm(M, N, K, A, lda, offa, B, ldb, offb, BIsSigned, C, ldc, CScale, Bias);

        for (size_t f = 0; f < M * N; f++) {
            // Sensitive to comparing positive/negative zero.
            if (C[f] != CReference[f]) {
                printf("mismatch M=%zd, N=%zd, K=%zd, offa=%d, offb=%d! %f %f\n", M, N, K, int(offa), int(offb), C[f], CReference[f]);
                break;
            }
        }
    }

    template<typename qint8_t>
    void
    DequantizeLinear(
        const qint8_t* Input,
        float* Output,
        size_t N,
        float scale,
        qint8_t offset
        )
    {
        for (size_t n = 0; n < N; n++) {
            Output[n] = float((int32_t(Input[n]) - offset)) * scale;
        }
    }

    MatrixGuardBuffer<uint8_t> BufferA;
    MatrixGuardBuffer<uint8_t> BufferB;
    MatrixGuardBuffer<float> BufferAFloat;
    MatrixGuardBuffer<float> BufferBFloat;
    MatrixGuardBuffer<float> BufferC;
    MatrixGuardBuffer<float> BufferCReference;
    MatrixGuardBuffer<float> BufferBias;
    const bool BIsSigned = std::is_signed<xint8_t>::value;

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (size_t b = 1; b < 16; b++) {
            Test(b, b, b, 34, 46);
        }
        for (size_t b = 16; b <= 256; b <<= 1) {
            Test(b, b, b, 15, 191);
        }
        for (size_t b = 256; b < 320; b += 32) {
            Test(b, b, b, 223, 73);
        }
        for (size_t b = 1; b < 96; b++) {
            Test(1, b, 32, 0, 0);
        }
        Test(43, 503, 401, 183, 223);
        Test(1024, 1024, 256, 13, 15);
    }
};

class MlasConv2DTest : public MlasTestBase
{
protected:
    void
    Test(
        size_t BatchCount,
        size_t GroupCount,
        size_t InputChannels,
        size_t InputHeight,
        size_t InputWidth,
        size_t FilterCount,
        size_t KernelHeight,
        size_t KernelWidth,
        size_t PaddingLeftHeight,
        size_t PaddingLeftWidth,
        size_t PaddingRightHeight,
        size_t PaddingRightWidth,
        size_t DilationHeight,
        size_t DilationWidth,
        size_t StrideHeight,
        size_t StrideWidth
        )
    {
        int64_t OutputHeight64 =
            ((int64_t(InputHeight) + int64_t(PaddingLeftHeight) + int64_t(PaddingRightHeight)) -
            (int64_t(DilationHeight) * (int64_t(KernelHeight) - 1) + 1)) / int64_t(StrideHeight) + 1;
        int64_t OutputWidth64 =
            ((int64_t(InputWidth) + int64_t(PaddingLeftWidth) + int64_t(PaddingRightWidth)) -
            (int64_t(DilationWidth) * (int64_t(KernelWidth) - 1) + 1)) / int64_t(StrideWidth) + 1;

        if (OutputHeight64 <= 0 || OutputWidth64 <= 0) {
            return;
        }

        size_t OutputHeight = size_t(OutputHeight64);
        size_t OutputWidth = size_t(OutputWidth64);

        size_t InputSize = InputHeight * InputWidth;
        size_t KernelSize = KernelHeight * KernelWidth;
        size_t OutputSize = OutputHeight * OutputWidth;

        size_t InputElements = BatchCount * GroupCount * InputChannels * InputSize;
        size_t FilterElements = GroupCount * FilterCount * InputChannels * KernelSize;
        size_t BiasElements = GroupCount * FilterCount;
        size_t OutputElements = BatchCount * GroupCount * FilterCount * OutputSize;

        const float* Input = BufferInput.GetBuffer(InputElements);
        const float* Filter = BufferFilter.GetBuffer(FilterElements);
        const float* Bias = BufferBias.GetBuffer(BiasElements);
        float* Output = BufferOutput.GetBuffer(OutputElements);
        float* OutputReference = BufferOutputReference.GetBuffer(OutputElements);

        MlasConv2D(BatchCount,
                   GroupCount,
                   InputChannels,
                   InputHeight, InputWidth,
                   FilterCount,
                   KernelHeight, KernelWidth,
                   PaddingLeftHeight, PaddingLeftWidth,
                   PaddingRightHeight, PaddingRightWidth,
                   DilationHeight, DilationWidth,
                   StrideHeight, StrideWidth,
                   OutputHeight, OutputWidth,
                   Input,
                   Filter,
                   Bias,
                   Output);

        ReferenceConv2D(BatchCount,
                        GroupCount,
                        InputChannels,
                        InputHeight, InputWidth,
                        FilterCount,
                        KernelHeight, KernelWidth,
                        PaddingLeftHeight, PaddingLeftWidth,
                        DilationHeight, DilationWidth,
                        StrideHeight, StrideWidth,
                        OutputHeight, OutputWidth,
                        Input,
                        Filter,
                        Bias,
                        OutputReference);

        if (memcmp(Output, OutputReference, OutputElements * sizeof(float)) != 0) {
            printf("mismatch: batch=%zd,group=%zd,input(%zd,%zd,%zd),filter=%zd,kernel(%zd,%zd)!!!\n",
                BatchCount, GroupCount, InputChannels, InputHeight, InputWidth, FilterCount,
                KernelHeight, KernelWidth);
        }
    }

    virtual
    void
    MlasConv2D(
        size_t BatchCount,
        size_t GroupCount,
        size_t InputChannels,
        size_t InputHeight,
        size_t InputWidth,
        size_t FilterCount,
        size_t KernelHeight,
        size_t KernelWidth,
        size_t PaddingLeftHeight,
        size_t PaddingLeftWidth,
        size_t PaddingRightHeight,
        size_t PaddingRightWidth,
        size_t DilationHeight,
        size_t DilationWidth,
        size_t StrideHeight,
        size_t StrideWidth,
        size_t OutputHeight,
        size_t OutputWidth,
        const float* Input,
        const float* Filter,
        const float* Bias,
        float* Output
        )
    {
        int64_t InputShape[] = { int64_t(InputHeight), int64_t(InputWidth) };
        int64_t KernelShape[] = { int64_t(KernelHeight), int64_t(KernelWidth) };
        int64_t DilationShape[] = { int64_t(DilationHeight), int64_t(DilationWidth) };
        int64_t Padding[] = { int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth) };
        int64_t StrideShape[] = { int64_t(StrideHeight), int64_t(StrideWidth) };
        int64_t OutputShape[] = { int64_t(OutputHeight), int64_t(OutputWidth) };

        MLAS_ACTIVATION Activation;
        Activation.ActivationKind = MlasIdentityActivation;

        MLAS_CONV_PARAMETERS Parameters;
        size_t WorkingBufferSize;

        MlasConvPrepare(&Parameters,
                        2,
                        BatchCount,
                        GroupCount,
                        InputChannels,
                        InputShape,
                        KernelShape,
                        DilationShape,
                        Padding,
                        StrideShape,
                        OutputShape,
                        FilterCount,
                        &Activation,
                        &WorkingBufferSize,
                        nullptr);

        MlasConv(&Parameters,
                 Input,
                 Filter,
                 Bias,
                 BufferWorking.GetBuffer(WorkingBufferSize),
                 Output,
                 nullptr);
    }

    void
    ReferenceConv2D(
        size_t BatchCount,
        size_t GroupCount,
        size_t InputChannels,
        size_t InputHeight,
        size_t InputWidth,
        size_t FilterCount,
        size_t KernelHeight,
        size_t KernelWidth,
        size_t PaddingLeftHeight,
        size_t PaddingLeftWidth,
        size_t DilationHeight,
        size_t DilationWidth,
        size_t StrideHeight,
        size_t StrideWidth,
        size_t OutputHeight,
        size_t OutputWidth,
        const float* Input,
        const float* Filter,
        const float* Bias,
        float* Output
        )
    {
        size_t InputSize = InputHeight * InputWidth;
        size_t OutputSize = OutputHeight * OutputWidth;
        size_t KernelSize = KernelHeight * KernelWidth;

        size_t K = InputChannels * KernelSize;
        size_t Im2ColElements = OutputSize * K;

        for (size_t b = 0; b < BatchCount; b++) {

            const float* filter = Filter;
            const float* bias = Bias;

            for (size_t g = 0; g < GroupCount; g++) {

                //
                // Transform the image using IM2COL and invoke the GEMM.
                //

                float* Im2Col = BufferIm2Col.GetBuffer(Im2ColElements);
                float* Im2ColOut = Im2Col;

                for (size_t c = 0; c < InputChannels; c++) {

                    for (size_t ky = 0; ky < KernelHeight; ky++) {

                        for (size_t kx = 0; kx < KernelWidth; kx++) {

                            for (size_t oh = 0; oh < OutputHeight; oh++) {

                                size_t ih = oh * StrideHeight + ky * DilationHeight - PaddingLeftHeight;

                                for (size_t ow = 0; ow < OutputWidth; ow++) {

                                    size_t iw = ow * StrideWidth + kx * DilationWidth - PaddingLeftWidth;

                                    *Im2ColOut++ = (ih < InputHeight && iw < InputWidth) ?
                                        Input[ih * InputWidth + iw] : 0;
                                }
                            }
                        }
                    }

                    Input += InputSize;
                }

                MlasGemm(CblasNoTrans, CblasNoTrans, FilterCount, OutputSize, K, 1.0f,
                    filter, K, Im2Col, OutputSize, 0.0f, Output, OutputSize, threadpool);

                //
                // Apply the bias.
                //

                for (size_t f = 0; f < FilterCount; f++) {

                    float biasValue = *bias++;

                    for (size_t o = 0; o < OutputSize; o++) {
                        *Output++ += biasValue;
                    }
                }

                filter += FilterCount * InputChannels * KernelSize;
            }
        }
    }

    MatrixGuardBuffer<float> BufferInput;
    MatrixGuardBuffer<float> BufferFilter;
    MatrixGuardBuffer<float> BufferBias;
    MatrixGuardBuffer<float> BufferOutput;
    MatrixGuardBuffer<float> BufferOutputReference;
    MatrixGuardBuffer<float> BufferWorking;
    MatrixGuardBuffer<float> BufferIm2Col;

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (unsigned i = 1; i < 256; i <<= 1) {
            Test(1, 1, 16, i, i, 32, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(1, 1, 16, i, i, 32, 3, 3, 0, 0, 0, 0, 1, 1, 2, 2);
            Test(1, 1, 16, i, i, 32, 3, 3, 0, 0, 0, 0, 2, 2, 1, 1);
            Test(1, 1, 16, i, i, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
            Test(1, 1, 16, i, i, 32, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(1, 1, 16, i, i, 32, i, 1, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(1, 1, 16, i, i, 32, 1, i, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(1, 16, 1, i, i, 1, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(1, 16, 1, i, i, 1, 3, 3, 0, 0, 0, 0, 1, 1, 2, 2);
            Test(1, 16, 1, i, i, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
            Test(1, 16, 1, i, i, 1, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2);
        }
    }

    void
    ExecuteLong(
        void
        ) override
    {
        static const unsigned cs[] = { 32, 14, 1 };
        static const unsigned is[] = { 53, 11, 5, 1 };

        for (unsigned i = 1; i <= 32; i++) {
            Test(4, 18, 1, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(4, 18, 1, 32, 89, 48, i, 89, 1, 1, 1, 1, 1, 1, 1, 1);
            Test(4, 18, 2, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
        }

        for (unsigned b = 1; b < 64; b++) {
            Test(b, 1, 64, 11, 11, 128, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
        }

        for (unsigned gc = 0; gc < _countof(cs); gc++) {
            for (unsigned ih = 0; ih < _countof(is); ih++) {
                for (unsigned iw = 0; iw < _countof(is); iw++) {
                    fprintf(stderr, "Handling depthwise %ux%ux%u\n", cs[gc], is[ih], is[iw]);
                    for (unsigned p0 = 0; p0 < 2; p0++) {
                        for (unsigned p1 = 0; p1 < 2; p1++) {
                            for (unsigned p2 = 0; p2 < 2; p2++) {
                                for (unsigned p3 = 0; p3 < 2; p3++) {
                                    for (unsigned dh = 1; dh <= 2; dh++) {
                                        for (unsigned dw = 1; dw <= 2; dw++) {
                                            for (unsigned sh = 1; sh <= 2; sh++) {
                                                for (unsigned sw = 1; sw <= 2; sw++) {
                                                    Test(1, cs[gc], 1, is[ih], is[iw], 1, 3, 3, p0, p1, p2, p3, dh, dw, sh, sw);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (unsigned ic = 0; ic < _countof(cs); ic++) {
            for (unsigned ih = 0; ih < _countof(is); ih++) {
                for (unsigned iw = 0; iw < _countof(is); iw++) {
                    fprintf(stderr, "Handling %ux%ux%u\n", cs[ic], is[ih], is[iw]);
                    for (unsigned fc = 0; fc < _countof(cs); fc++) {
                        for (unsigned kh = 1; kh <= 5; kh++) {
                            if (kh == 4) continue;
                            for (unsigned kw = 1; kw <= 5; kw++) {
                                if (kw == 4) continue;
                                for (unsigned p0 = 0; p0 < 2; p0++) {
                                    for (unsigned p1 = 0; p1 < 2; p1++) {
                                        for (unsigned p2 = 0; p2 < 2; p2++) {
                                            for (unsigned p3 = 0; p3 < 2; p3++) {
                                                for (unsigned dh = 1; dh <= 2; dh++) {
                                                    for (unsigned dw = 1; dw <= 2; dw++) {
                                                        for (unsigned sh = 1; sh <= 2; sh++) {
                                                            for (unsigned sw = 1; sw <= 2; sw++) {
                                                                Test(1, 1, cs[ic], is[ih], is[iw], cs[fc], kh, kw, p0, p1, p2, p3, dh, dw, sh, sw);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

class MlasNchwcConv2DTest : public MlasConv2DTest
{
protected:
    void
    MlasConv2D(
        size_t BatchCount,
        size_t GroupCount,
        size_t InputChannels,
        size_t InputHeight,
        size_t InputWidth,
        size_t FilterCount,
        size_t KernelHeight,
        size_t KernelWidth,
        size_t PaddingLeftHeight,
        size_t PaddingLeftWidth,
        size_t PaddingRightHeight,
        size_t PaddingRightWidth,
        size_t DilationHeight,
        size_t DilationWidth,
        size_t StrideHeight,
        size_t StrideWidth,
        size_t OutputHeight,
        size_t OutputWidth,
        const float* Input,
        const float* Filter,
        const float* Bias,
        float* Output
        ) override
    {
        int64_t InputShape[] = { int64_t(BatchCount), int64_t(GroupCount * InputChannels), int64_t(InputHeight), int64_t(InputWidth) };
        int64_t FilterShape[] = { int64_t(GroupCount * FilterCount), int64_t(InputChannels), int64_t(KernelHeight), int64_t(KernelWidth) };
        int64_t OutputShape[] = { int64_t(BatchCount), int64_t(GroupCount * FilterCount), int64_t(OutputHeight), int64_t(OutputWidth) };

        int64_t KernelShape[] = { int64_t(KernelHeight), int64_t(KernelWidth) };
        int64_t DilationShape[] = { int64_t(DilationHeight), int64_t(DilationWidth) };
        int64_t Padding[] = { int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth) };
        int64_t StrideShape[] = { int64_t(StrideHeight), int64_t(StrideWidth) };

        //
        // Select the type of convolution that will be performed.
        //

        bool DoReorderInput;
        bool ReorderFilterOIHWBo;

        if (GroupCount > 1 && InputChannels == 1 && FilterCount == 1) {
            // Depthwise convolution.
            DoReorderInput = true;
            ReorderFilterOIHWBo = true;
        } else if (InputChannels >= BlockSize) {
            // NCHWc or pointwise convolution;
            DoReorderInput = true;
            ReorderFilterOIHWBo = false;
        } else {
            // NCHW convolution.
            DoReorderInput = false;
            ReorderFilterOIHWBo = true;
        }

        size_t NchwcInputChannels = (GroupCount * InputChannels + BlockSize - 1) & ~(BlockSize - 1);
        size_t NchwcOutputChannels = (GroupCount * FilterCount + BlockSize - 1) & ~(BlockSize - 1);

        //
        // Reorder the filter buffer as needed.
        //

        float* ReorderedFilter;

        if (ReorderFilterOIHWBo) {
            size_t NchwcFilterElements = NchwcOutputChannels * InputChannels * KernelHeight * KernelWidth;
            ReorderedFilter = BufferNchwcFilter.GetBuffer(NchwcFilterElements);
            MlasReorderFilterOIHWBo(FilterShape, Filter, ReorderedFilter);
        } else {
            size_t NchwcFilterElements = NchwcOutputChannels * NchwcInputChannels * KernelHeight * KernelWidth;
            ReorderedFilter = BufferNchwcFilter.GetBuffer(NchwcFilterElements);
            MlasReorderFilterOIHWBiBo(FilterShape, Filter, ReorderedFilter);
        }

        //
        // Align the bias buffer to the filter count if needed.
        //

        if (Bias != nullptr && GroupCount * FilterCount < NchwcOutputChannels) {

            float* AlignedBias = BufferNchwcBias.GetBuffer(NchwcOutputChannels);

            size_t i;
            for (i = 0; i < GroupCount * FilterCount; i++) {
                AlignedBias[i] = Bias[i];
            }
            for (; i < NchwcOutputChannels; i++) {
                AlignedBias[i] = 0.0f;
            }

            Bias = AlignedBias;
        }

        //
        // Reorder the input buffer if needed.
        //

        if (DoReorderInput) {
            size_t NchwcInputElements = BatchCount * NchwcInputChannels * InputHeight * InputWidth;
            float* NchwcInput = BufferNchwcInput.GetBuffer(NchwcInputElements);
            MlasReorderInput(InputShape, Input, NchwcInput);
            Input = NchwcInput;
            InputShape[1] = NchwcInputChannels;
        }

        int64_t NchwcOutputShape[] = { int64_t(BatchCount), int64_t(NchwcOutputChannels), int64_t(OutputHeight), int64_t(OutputWidth) };

        size_t NchwcOutputElements = BatchCount * NchwcOutputChannels * OutputHeight * OutputWidth;
        float* NchwcOutput = BufferNchwcOutput.GetBuffer(NchwcOutputElements);

        MLAS_ACTIVATION Activation;
        Activation.ActivationKind = MlasIdentityActivation;

        MlasNchwcConv(InputShape,
                      KernelShape,
                      DilationShape,
                      Padding,
                      StrideShape,
                      NchwcOutputShape,
                      GroupCount,
                      Input,
                      ReorderedFilter,
                      Bias,
                      NchwcOutput,
                      &Activation,
                      true,
                      nullptr);

        //
        // Reorder the output buffer.
        //

        MlasReorderOutputNchw(OutputShape, NchwcOutput, Output);
    }

    const size_t BlockSize = MlasNchwcGetBlockSize();

    MatrixGuardBuffer<float> BufferNchwcInput;
    MatrixGuardBuffer<float> BufferNchwcFilter;
    MatrixGuardBuffer<float> BufferNchwcBias;
    MatrixGuardBuffer<float> BufferNchwcOutput;

public:
    void
    ExecuteLong(
        void
        ) override
    {
        // N.B. InputChannels must be a multiple of 4 if the count is greater
        // than the block size.
        static const unsigned cis[] = { 32, 20, 5, 1 };
        static const unsigned cos[] = { 64, 15, 1 };
        static const unsigned is[] = { 27, 11, 5, 1 };

        // Depthwise convolutions.
        for (unsigned i = 16; i < 256; i <<= 1) {
            Test(1, i, 1, 28, 28, 1, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(1, i, 1, 28, 28, 1, 3, 3, 0, 0, 0, 0, 1, 1, 2, 2);
            Test(1, i, 1, 28, 28, 1, 3, 3, 0, 0, 0, 0, 2, 2, 1, 1);
            Test(1, i, 1, 28, 28, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
            Test(1, i, 1, 28, 28, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(1, i, 1, 28, 28, 1, i, 1, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(12, i, 1, 11, 11, 1, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
        }

        // Test varying FilterCounts.
        for (unsigned i = 1; i < 128; i++) {
            Test(1, 1, 3, 34, 34, i, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(1, 1, 16, 34, 34, i, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(1, 1, 16, 34, 34, i, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
        }

        for (unsigned i = 1; i <= 32; i++) {
            Test(4, 18, 1, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(4, 18, 1, 32, 89, 48, i, 89, 1, 1, 1, 1, 1, 1, 1, 1);
            Test(4, 18, 2, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
        }

        for (unsigned b = 1; b < 64; b++) {
            Test(b, 1, 64, 11, 11, 128, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
        }

        for (unsigned ic = 0; ic < _countof(cis); ic++) {
            for (unsigned ih = 0; ih < _countof(is); ih++) {
                for (unsigned iw = 0; iw < _countof(is); iw++) {
                    fprintf(stderr, "Handling %ux%ux%u\n", cis[ic], is[ih], is[iw]);
                    for (unsigned fc = 0; fc < _countof(cos); fc++) {
                        for (unsigned kh = 1; kh <= 5; kh++) {
                            if (kh == 4) continue;
                            for (unsigned kw = 1; kw <= 5; kw++) {
                                if (kw == 4) continue;
                                for (unsigned p0 = 0; p0 <= 3; p0++) {
                                    for (unsigned p1 = 0; p1 <= 3; p1++) {
                                        for (unsigned p2 = 0; p2 <= 3; p2++) {
                                            for (unsigned p3 = 0; p3 <= 3; p3++) {
                                                for (unsigned dh = 1; dh <= 2; dh++) {
                                                    for (unsigned dw = 1; dw <= 2; dw++) {
                                                        for (unsigned sh = 1; sh <= 2; sh++) {
                                                            for (unsigned sw = 1; sw <= 2; sw++) {
                                                                Test(1, 1, cis[ic], is[ih], is[iw], cos[fc], kh, kw, p0, p1, p2, p3, dh, dw, sh, sw);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

};

class MlasPool2DTest : public MlasTestBase
{
protected:
    void
    Test(
        size_t BatchCount,
        size_t InputChannels,
        size_t InputHeight,
        size_t InputWidth,
        size_t KernelHeight,
        size_t KernelWidth,
        size_t PaddingLeftHeight,
        size_t PaddingLeftWidth,
        size_t PaddingRightHeight,
        size_t PaddingRightWidth,
        size_t StrideHeight,
        size_t StrideWidth
        )
    {
        const size_t DilationHeight = 1;
        const size_t DilationWidth = 1;

        int64_t OutputHeight64 =
            ((int64_t(InputHeight) + int64_t(PaddingLeftHeight) + int64_t(PaddingRightHeight)) -
            (int64_t(DilationHeight) * (int64_t(KernelHeight) - 1) + 1)) / int64_t(StrideHeight) + 1;
        int64_t OutputWidth64 =
            ((int64_t(InputWidth) + int64_t(PaddingLeftWidth) + int64_t(PaddingRightWidth)) -
            (int64_t(DilationWidth) * (int64_t(KernelWidth) - 1) + 1)) / int64_t(StrideWidth) + 1;

        if (OutputHeight64 <= 0 || OutputWidth64 <= 0) {
            return;
        }

        int64_t InputShape[] = { int64_t(BatchCount), int64_t(InputChannels), int64_t(InputHeight), int64_t(InputWidth) };
        int64_t KernelShape[] = { int64_t(KernelHeight), int64_t(KernelWidth) };
        int64_t Padding[] = { int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth) };
        int64_t StrideShape[] = { int64_t(StrideHeight), int64_t(StrideWidth) };
        int64_t OutputShape[] = { int64_t(BatchCount), int64_t(InputChannels), OutputHeight64, OutputWidth64 };

        size_t InputBufferElements = size_t(InputShape[0] * InputShape[1] * InputShape[2] * InputShape[3]);
        size_t OutputBufferElements = size_t(OutputShape[0] * OutputShape[1] * OutputShape[2] * OutputShape[3]);

        const float* Input = BufferInput.GetBuffer(InputBufferElements);
        float* Output = BufferOutput.GetBuffer(OutputBufferElements);
        float* OutputReference = BufferOutputReference.GetBuffer(OutputBufferElements);

        MlasPool2D(MlasMaximumPooling, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output);
        ReferenceMaximumPool2D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference);

        if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
            printf("mismatch: maximum input(%zd,%zd,%zd),kernel(%zd,%zd)!!!\n",
                InputChannels, InputHeight, InputWidth, KernelHeight, KernelWidth);
        }

        MlasPool2D(MlasAveragePoolingExcludePad, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output);
        ReferenceAveragePool2D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference, false);

        if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
            printf("mismatch: averageexcpad input(%zd,%zd,%zd),kernel(%zd,%zd)!!!\n",
                InputChannels, InputHeight, InputWidth, KernelHeight, KernelWidth);
        }

        MlasPool2D(MlasAveragePoolingIncludePad, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output);
        ReferenceAveragePool2D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference, true);

        if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
            printf("mismatch: averageincpad input(%zd,%zd,%zd),kernel(%zd,%zd)!!!\n",
                InputChannels, InputHeight, InputWidth, KernelHeight, KernelWidth);
        }
    }

    virtual
    void
    MlasPool2D(
        MLAS_POOLING_KIND PoolingKind,
        const int64_t* InputShape,
        const int64_t* KernelShape,
        const int64_t* Padding,
        const int64_t* StrideShape,
        const int64_t* OutputShape,
        const float* Input,
        float* Output
        )
    {
        MlasPool(PoolingKind, 2, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output, threadpool);
    }

    void
    ReferenceMaximumPool2D(
        const int64_t* InputShape,
        const int64_t* KernelShape,
        const int64_t* Padding,
        const int64_t* StrideShape,
        const float* Input,
        float* Output
        )
    {
        int64_t ChannelCount = InputShape[0] * InputShape[1];

        int64_t InputHeight = InputShape[2];
        int64_t InputWidth = InputShape[3];

        int64_t KernelHeight = KernelShape[0];
        int64_t KernelWidth = KernelShape[1];

        int64_t PaddingLeftY = Padding[0];
        int64_t PaddingLeftX = Padding[1];
        int64_t PaddingRightY = Padding[2];
        int64_t PaddingRightX = Padding[3];

        int64_t StrideHeight = StrideShape[0];
        int64_t StrideWidth = StrideShape[1];

        int64_t OutputHeight = (InputHeight + PaddingLeftY + PaddingRightY - KernelHeight) / StrideHeight + 1;
        int64_t OutputWidth = (InputWidth + PaddingLeftX + PaddingRightX - KernelWidth) / StrideWidth + 1;

        for (int64_t c = 0; c < ChannelCount; c++) {

            for (int64_t ph = 0; ph < OutputHeight; ph++) {

                int64_t ihStart = ph * StrideHeight - PaddingLeftY;
                int64_t ihEnd = ihStart + KernelHeight;

                ihStart = (std::max)(ihStart, int64_t(0));
                ihEnd = (std::min)(ihEnd, InputHeight);

                for (int64_t pw = 0; pw < OutputWidth; pw++) {

                    int64_t iwStart = pw * StrideWidth - PaddingLeftX;
                    int64_t iwEnd = iwStart + KernelWidth;

                    iwStart = (std::max)(iwStart, int64_t(0));
                    iwEnd = (std::min)(iwEnd, InputWidth);

                    float m = std::numeric_limits<float>::lowest();

                    for (int64_t ih = ihStart; ih < ihEnd; ih++) {
                        for (int64_t iw = iwStart; iw < iwEnd; iw++) {
                            m = (std::max)(m, Input[ih * InputWidth + iw]);
                        }
                    }

                    Output[ph * OutputWidth + pw] = m;
                }
            }

            Input += InputHeight * InputWidth;
            Output += OutputHeight * OutputWidth;
        }
    }

    void
    ReferenceAveragePool2D(
        const int64_t* InputShape,
        const int64_t* KernelShape,
        const int64_t* Padding,
        const int64_t* StrideShape,
        const float* Input,
        float* Output,
        bool CountIncludePad
        )
    {
        int64_t ChannelCount = InputShape[0] * InputShape[1];

        int64_t InputHeight = InputShape[2];
        int64_t InputWidth = InputShape[3];

        int64_t KernelHeight = KernelShape[0];
        int64_t KernelWidth = KernelShape[1];

        int64_t PaddingLeftY = Padding[0];
        int64_t PaddingLeftX = Padding[1];
        int64_t PaddingRightY = Padding[2];
        int64_t PaddingRightX = Padding[3];

        int64_t StrideHeight = StrideShape[0];
        int64_t StrideWidth = StrideShape[1];

        int64_t OutputHeight = (InputHeight + PaddingLeftY + PaddingRightY - KernelHeight) / StrideHeight + 1;
        int64_t OutputWidth = (InputWidth + PaddingLeftX + PaddingRightX - KernelWidth) / StrideWidth + 1;

        for (int64_t c = 0; c < ChannelCount; c++) {

            for (int64_t ph = 0; ph < OutputHeight; ph++) {

                int64_t ihStart = ph * StrideHeight - PaddingLeftY;
                int64_t ihEnd = ihStart + KernelHeight;

                ihStart = (std::max)(ihStart, int64_t(0));
                ihEnd = (std::min)(ihEnd, InputHeight);

                for (int64_t pw = 0; pw < OutputWidth; pw++) {

                    int64_t iwStart = pw * StrideWidth - PaddingLeftX;
                    int64_t iwEnd = iwStart + KernelWidth;

                    iwStart = (std::max)(iwStart, int64_t(0));
                    iwEnd = (std::min)(iwEnd, InputWidth);

                    float m = 0.0f;

                    for (int64_t ih = ihStart; ih < ihEnd; ih++) {
                        for (int64_t iw = iwStart; iw < iwEnd; iw++) {
                            m += Input[ih * InputWidth + iw];
                        }
                    }

                    if (CountIncludePad) {
                        m /= (KernelHeight * KernelWidth);
                    } else {
                        m /= (ihEnd - ihStart) * (iwEnd - iwStart);
                    }

                    Output[ph * OutputWidth + pw] = m;
                }
            }

            Input += InputHeight * InputWidth;
            Output += OutputHeight * OutputWidth;
        }
    }

    MatrixGuardBuffer<float> BufferInput;
    MatrixGuardBuffer<float> BufferOutput;
    MatrixGuardBuffer<float> BufferOutputReference;

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (unsigned i = 1; i < 256; i <<= 1) {
            Test(1, 16, i, i, 3, 3, 0, 0, 0, 0, 1, 1);
            Test(1, 16, i, i, 3, 3, 0, 0, 0, 0, 2, 2);
            Test(1, 16, i, i, 3, 3, 0, 0, 0, 0, 1, 1);
            Test(1, 16, i, i, 3, 3, 1, 1, 1, 1, 1, 1);
            Test(1, 16, i, i, 1, 1, 0, 0, 0, 0, 1, 1);
            Test(1, 16, i, i, i, 1, 0, 0, 0, 0, 1, 1);
            Test(1, 16, i, i, 1, i, 0, 0, 0, 0, 1, 1);
        }
    }

    void
    ExecuteLong(
        void
        ) override
    {
        static const unsigned is[] = { 53, 17, 11, 5, 4, 3, 2, 1 };

        for (unsigned i = 1; i < 2058; i++) {
            Test(1, 1, 4, i, 2, 4, 0, 2, 0, 1, 1, 1);
        }

        for (unsigned ih = 0; ih < _countof(is); ih++) {
            for (unsigned iw = 0; iw < _countof(is); iw++) {
                fprintf(stderr, "Handling %ux%u\n", is[ih], is[iw]);
                Test(1, 1, is[ih], is[iw], is[ih], is[iw], 0, 0, 0, 0, 1, 1);
                Test(1, 1, is[ih], is[iw], is[ih], 1, 0, 0, 0, 0, 1, 1);
                Test(1, 1, is[ih], is[iw], 1, is[iw], 0, 0, 0, 0, 1, 1);
                for (unsigned kh = 1; kh <= 5; kh++) {
                    if (kh > is[ih]) break;
                    for (unsigned kw = 1; kw <= 5; kw++) {
                        if (kw > is[iw]) break;
                        for (unsigned sh = 1; sh <= 3; sh++) {
                            for (unsigned sw = 1; sw <= 3; sw++) {
                                for (unsigned p0 = 0; p0 < kh; p0++) {
                                    for (unsigned p1 = 0; p1 < kw; p1++) {
                                        for (unsigned p2 = 0; p2 < kh; p2++) {
                                            for (unsigned p3 = 0; p3 < kw; p3++) {
                                                Test(5, 3, is[ih], is[iw], kh, kw, p0, p1, p2, p3, sh, sw);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

class MlasNchwcPool2DTest : public MlasPool2DTest
{
protected:
    void
    MlasPool2D(
        MLAS_POOLING_KIND PoolingKind,
        const int64_t* InputShape,
        const int64_t* KernelShape,
        const int64_t* Padding,
        const int64_t* StrideShape,
        const int64_t* OutputShape,
        const float* Input,
        float* Output
        ) override
    {
        size_t NchwcChannels = (size_t(InputShape[1]) + BlockSize - 1) & ~(BlockSize - 1);

        int64_t NchwcInputShape[] = { InputShape[0], int64_t(NchwcChannels), InputShape[2], InputShape[3] };
        size_t NchwcInputElements = size_t(NchwcInputShape[0]) * size_t(NchwcInputShape[1]) * size_t(NchwcInputShape[2]) * size_t(NchwcInputShape[3]);
        float* NchwcInput = BufferNchwcInput.GetBuffer(NchwcInputElements);

        int64_t NchwcOutputShape[] = { OutputShape[0], int64_t(NchwcChannels), OutputShape[2], OutputShape[3] };
        size_t NchwcOutputElements = size_t(NchwcOutputShape[0]) * size_t(NchwcOutputShape[1]) * size_t(NchwcOutputShape[2]) * size_t(NchwcOutputShape[3]);
        float* NchwcOutput = BufferNchwcOutput.GetBuffer(NchwcOutputElements);

        MlasReorderInput(InputShape, Input, NchwcInput);

        MlasNchwcPool(PoolingKind,
                      NchwcInputShape,
                      KernelShape,
                      nullptr,
                      Padding,
                      StrideShape,
                      NchwcOutputShape,
                      NchwcInput,
                      NchwcOutput,
                      nullptr);

        MlasReorderOutputNchw(OutputShape, NchwcOutput, Output);
    }

    MatrixGuardBuffer<float> BufferNchwcInput;
    MatrixGuardBuffer<float> BufferNchwcOutput;

    const size_t BlockSize = MlasNchwcGetBlockSize();

public:
    void
    ExecuteLong(
        void
        ) override
    {
        static const unsigned is[] = { 53, 11, 1 };

        for (unsigned ih = 0; ih < _countof(is); ih++) {
            for (unsigned iw = 0; iw < _countof(is); iw++) {
                fprintf(stderr, "Handling %ux%u\n", is[ih], is[iw]);
                Test(1, 12, is[ih], is[iw], is[ih], is[iw], 0, 0, 0, 0, 1, 1);
                Test(1, 32, is[ih], is[iw], is[ih], 1, 0, 0, 0, 0, 1, 1);
                Test(1, 68, is[ih], is[iw], 1, is[iw], 0, 0, 0, 0, 1, 1);
                for (unsigned kh = 1; kh <= 5; kh++) {
                    if (kh > is[ih]) break;
                    for (unsigned kw = 1; kw <= 5; kw++) {
                        if (kw > is[iw]) break;
                        for (unsigned sh = 1; sh <= 3; sh++) {
                            for (unsigned sw = 1; sw <= 3; sw++) {
                                for (unsigned p0 = 0; p0 < kh; p0++) {
                                    for (unsigned p1 = 0; p1 < kw; p1++) {
                                        for (unsigned p2 = 0; p2 < kh; p2++) {
                                            for (unsigned p3 = 0; p3 < kw; p3++) {
                                                Test(1, 32, is[ih], is[iw], kh, kw, p0, p1, p2, p3, sh, sw);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

};

class MlasPool3DTest : public MlasTestBase
{
protected:
    void
    Test(
        size_t BatchCount,
        size_t InputChannels,
        size_t InputDepth,
        size_t InputHeight,
        size_t InputWidth,
        size_t KernelDepth,
        size_t KernelHeight,
        size_t KernelWidth,
        size_t PaddingLeftDepth,
        size_t PaddingLeftHeight,
        size_t PaddingLeftWidth,
        size_t PaddingRightDepth,
        size_t PaddingRightHeight,
        size_t PaddingRightWidth,
        size_t StrideDepth,
        size_t StrideHeight,
        size_t StrideWidth
        )
    {
        const size_t DilationDepth = 1;
        const size_t DilationHeight = 1;
        const size_t DilationWidth = 1;

        int64_t OutputDepth64 =
            ((int64_t(InputDepth) + int64_t(PaddingLeftDepth) + int64_t(PaddingRightDepth)) -
            (int64_t(DilationDepth) * (int64_t(KernelDepth) - 1) + 1)) / int64_t(StrideDepth) + 1;
        int64_t OutputHeight64 =
            ((int64_t(InputHeight) + int64_t(PaddingLeftHeight) + int64_t(PaddingRightHeight)) -
            (int64_t(DilationHeight) * (int64_t(KernelHeight) - 1) + 1)) / int64_t(StrideHeight) + 1;
        int64_t OutputWidth64 =
            ((int64_t(InputWidth) + int64_t(PaddingLeftWidth) + int64_t(PaddingRightWidth)) -
            (int64_t(DilationWidth) * (int64_t(KernelWidth) - 1) + 1)) / int64_t(StrideWidth) + 1;

        if (OutputDepth64 <= 0 || OutputHeight64 <= 0 || OutputWidth64 <= 0) {
            return;
        }

        int64_t InputShape[] = { int64_t(BatchCount), int64_t(InputChannels), int64_t(InputDepth), int64_t(InputHeight), int64_t(InputWidth) };
        int64_t KernelShape[] = { int64_t(KernelDepth), int64_t(KernelHeight), int64_t(KernelWidth) };
        int64_t Padding[] = { int64_t(PaddingLeftDepth), int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightDepth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth) };
        int64_t StrideShape[] = { int64_t(StrideDepth), int64_t(StrideHeight), int64_t(StrideWidth) };
        int64_t OutputShape[] = { int64_t(BatchCount), int64_t(InputChannels), OutputDepth64, OutputHeight64, OutputWidth64 };

        OutputShape[2] = (InputShape[2] + Padding[0] + Padding[3] - KernelShape[0]) / StrideShape[0] + 1;
        OutputShape[3] = (InputShape[3] + Padding[1] + Padding[4] - KernelShape[1]) / StrideShape[1] + 1;
        OutputShape[4] = (InputShape[4] + Padding[2] + Padding[5] - KernelShape[2]) / StrideShape[2] + 1;

        size_t InputBufferElements = size_t(InputShape[0] * InputShape[1] * InputShape[2] * InputShape[3] * InputShape[4]);
        size_t OutputBufferElements = size_t(OutputShape[0] * OutputShape[1] * OutputShape[2] * OutputShape[3] * OutputShape[4]);

        const float* Input = BufferInput.GetBuffer(InputBufferElements);
        float* Output = BufferOutput.GetBuffer(OutputBufferElements);
        float* OutputReference = BufferOutputReference.GetBuffer(OutputBufferElements);

        MlasPool(MlasMaximumPooling, 3, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output, threadpool);
        ReferenceMaximumPool3D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference);

        if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
            printf("mismatch: maximum input(%zd,%zd,%zd,%zd),kernel(%zd,%zd,%zd)!!!\n",
                InputChannels, InputDepth, InputHeight, InputWidth, KernelDepth, KernelHeight, KernelWidth);
        }

        MlasPool(MlasAveragePoolingExcludePad, 3, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output, threadpool);
        ReferenceAveragePool3D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference, false);

        if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
            printf("mismatch: averageexcpad input(%zd,%zd,%zd,%zd),kernel(%zd,%zd,%zd)!!!\n",
                InputChannels, InputDepth, InputHeight, InputWidth, KernelDepth, KernelHeight, KernelWidth);
        }

        MlasPool(MlasAveragePoolingIncludePad, 3, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output, threadpool);
        ReferenceAveragePool3D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference, true);

        if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
            printf("mismatch: averageincpad input(%zd,%zd,%zd,%zd),kernel(%zd,%zd,%zd)!!!\n",
                InputChannels, InputDepth, InputHeight, InputWidth, KernelDepth, KernelHeight, KernelWidth);
        }
    }

    void
    ReferenceMaximumPool3D(
        const int64_t* InputShape,
        const int64_t* KernelShape,
        const int64_t* Padding,
        const int64_t* StrideShape,
        const float* Input,
        float* Output
        )
    {
        int64_t ChannelCount = InputShape[0] * InputShape[1];

        int64_t InputDepth = InputShape[2];
        int64_t InputHeight = InputShape[3];
        int64_t InputWidth = InputShape[4];

        int64_t KernelDepth = KernelShape[0];
        int64_t KernelHeight = KernelShape[1];
        int64_t KernelWidth = KernelShape[2];

        int64_t PaddingLeftZ = Padding[0];
        int64_t PaddingLeftY = Padding[1];
        int64_t PaddingLeftX = Padding[2];
        int64_t PaddingRightZ = Padding[3];
        int64_t PaddingRightY = Padding[4];
        int64_t PaddingRightX = Padding[5];

        int64_t StrideDepth = StrideShape[0];
        int64_t StrideHeight = StrideShape[1];
        int64_t StrideWidth = StrideShape[2];

        int64_t OutputDepth = (InputDepth + PaddingLeftZ + PaddingRightZ - KernelDepth) / StrideDepth + 1;
        int64_t OutputHeight = (InputHeight + PaddingLeftY + PaddingRightY - KernelHeight) / StrideHeight + 1;
        int64_t OutputWidth = (InputWidth + PaddingLeftX + PaddingRightX - KernelWidth) / StrideWidth + 1;

        for (int64_t c = 0; c < ChannelCount; c++) {

            for (int64_t pd = 0; pd < OutputDepth; pd++) {

                int64_t idStart = pd * StrideDepth - PaddingLeftZ;
                int64_t idEnd = idStart + KernelDepth;

                idStart = (std::max)(idStart, int64_t(0));
                idEnd = (std::min)(idEnd, InputDepth);

                for (int64_t ph = 0; ph < OutputHeight; ph++) {

                    int64_t ihStart = ph * StrideHeight - PaddingLeftY;
                    int64_t ihEnd = ihStart + KernelHeight;

                    ihStart = (std::max)(ihStart, int64_t(0));
                    ihEnd = (std::min)(ihEnd, InputHeight);

                    for (int64_t pw = 0; pw < OutputWidth; pw++) {

                        int64_t iwStart = pw * StrideWidth - PaddingLeftX;
                        int64_t iwEnd = iwStart + KernelWidth;

                        iwStart = (std::max)(iwStart, int64_t(0));
                        iwEnd = (std::min)(iwEnd, InputWidth);

                        float m = std::numeric_limits<float>::lowest();

                        for (int64_t id = idStart; id < idEnd; id++) {
                            for (int64_t ih = ihStart; ih < ihEnd; ih++) {
                                for (int64_t iw = iwStart; iw < iwEnd; iw++) {
                                    m = (std::max)(m, Input[id * InputHeight * InputWidth + ih * InputWidth + iw]);
                                }
                            }
                        }

                        Output[pd * OutputHeight * OutputWidth + ph * OutputWidth + pw] = m;
                    }
                }
            }

            Input += InputDepth * InputHeight * InputWidth;
            Output += OutputDepth * OutputHeight * OutputWidth;
        }
    }

    void
    ReferenceAveragePool3D(
        const int64_t* InputShape,
        const int64_t* KernelShape,
        const int64_t* Padding,
        const int64_t* StrideShape,
        const float* Input,
        float* Output,
        bool CountIncludePad
        )
    {
        int64_t ChannelCount = InputShape[0] * InputShape[1];

        int64_t InputDepth = InputShape[2];
        int64_t InputHeight = InputShape[3];
        int64_t InputWidth = InputShape[4];

        int64_t KernelDepth = KernelShape[0];
        int64_t KernelHeight = KernelShape[1];
        int64_t KernelWidth = KernelShape[2];

        int64_t PaddingLeftZ = Padding[0];
        int64_t PaddingLeftY = Padding[1];
        int64_t PaddingLeftX = Padding[2];
        int64_t PaddingRightZ = Padding[3];
        int64_t PaddingRightY = Padding[4];
        int64_t PaddingRightX = Padding[5];

        int64_t StrideDepth = StrideShape[0];
        int64_t StrideHeight = StrideShape[1];
        int64_t StrideWidth = StrideShape[2];

        int64_t OutputDepth = (InputDepth + PaddingLeftZ + PaddingRightZ - KernelDepth) / StrideDepth + 1;
        int64_t OutputHeight = (InputHeight + PaddingLeftY + PaddingRightY - KernelHeight) / StrideHeight + 1;
        int64_t OutputWidth = (InputWidth + PaddingLeftX + PaddingRightX - KernelWidth) / StrideWidth + 1;

        for (int64_t c = 0; c < ChannelCount; c++) {

            for (int64_t pd = 0; pd < OutputDepth; pd++) {

                int64_t idStart = pd * StrideDepth - PaddingLeftZ;
                int64_t idEnd = idStart + KernelDepth;

                idStart = (std::max)(idStart, int64_t(0));
                idEnd = (std::min)(idEnd, InputDepth);

                for (int64_t ph = 0; ph < OutputHeight; ph++) {

                    int64_t ihStart = ph * StrideHeight - PaddingLeftY;
                    int64_t ihEnd = ihStart + KernelHeight;

                    ihStart = (std::max)(ihStart, int64_t(0));
                    ihEnd = (std::min)(ihEnd, InputHeight);

                    for (int64_t pw = 0; pw < OutputWidth; pw++) {

                        int64_t iwStart = pw * StrideWidth - PaddingLeftX;
                        int64_t iwEnd = iwStart + KernelWidth;

                        iwStart = (std::max)(iwStart, int64_t(0));
                        iwEnd = (std::min)(iwEnd, InputWidth);

                        float m = 0.0f;

                        for (int64_t id = idStart; id < idEnd; id++) {
                            for (int64_t ih = ihStart; ih < ihEnd; ih++) {
                                for (int64_t iw = iwStart; iw < iwEnd; iw++) {
                                    m += Input[id * InputHeight * InputWidth + ih * InputWidth + iw];
                                }
                            }
                        }

                        if (CountIncludePad) {
                            m /= (KernelDepth * KernelHeight * KernelWidth);
                        } else {
                            m /= (idEnd - idStart) * (ihEnd - ihStart) * (iwEnd - iwStart);
                        }

                        Output[pd * OutputHeight * OutputWidth + ph * OutputWidth + pw] = m;
                    }
                }
            }

            Input += InputDepth * InputHeight * InputWidth;
            Output += OutputDepth * OutputHeight * OutputWidth;
        }
    }

    MatrixGuardBuffer<float> BufferInput;
    MatrixGuardBuffer<float> BufferOutput;
    MatrixGuardBuffer<float> BufferOutputReference;

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (unsigned i = 1; i < 64; i <<= 1) {
            Test(1, 16, i, i, i, 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1);
            Test(1, 16, i, i, i, 3, 3, 3, 0, 0, 0, 0, 0, 0, 2, 2, 2);
            Test(1, 16, i, i, i, 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1);
            Test(1, 16, i, i, i, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1);
            Test(1, 16, i, i, i, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1);
            Test(1, 16, i, i, i, 1, i, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1);
            Test(1, 16, i, i, i, 1, 1, i, 0, 0, 0, 0, 0, 0, 1, 1, 1);
        }
    }

    void
    ExecuteLong(
        void
        ) override
    {
        static const unsigned is[] = { 11, 5, 4, 3, 2, 1 };

        for (unsigned id = 0; id < _countof(is); id++) {
            for (unsigned ih = 0; ih < _countof(is); ih++) {
                for (unsigned iw = 0; iw < _countof(is); iw++) {
                    fprintf(stderr, "Handling %ux%ux%u\n", is[id], is[ih], is[iw]);
                    Test(1, 1, is[id], is[ih], is[iw], is[id], is[ih], is[iw], 0, 0, 0, 0, 0, 0, 1, 1, 1);
                    for (unsigned kd = 1; kd <= 4; kd++) {
                        if (kd > is[id]) break;
                        for (unsigned kh = 1; kh <= 4; kh++) {
                            if (kh > is[ih]) break;
                            for (unsigned kw = 1; kw <= 4; kw++) {
                                if (kw > is[iw]) break;
                                for (unsigned sd = 1; sd <= 3; sd++) {
                                    for (unsigned sh = 1; sh <= 3; sh++) {
                                        for (unsigned sw = 1; sw <= 3; sw++) {
                                            for (unsigned p0 = 0; p0 < kd; p0++) {
                                                for (unsigned p1 = 0; p1 < kh; p1++) {
                                                    for (unsigned p2 = 0; p2 < kw; p2++) {
                                                        for (unsigned p3 = 0; p3 < kd; p3++) {
                                                            for (unsigned p4 = 0; p4 < kh; p4++) {
                                                                for (unsigned p5 = 0; p5 < kw; p5++) {
                                                                    Test(1, 1, is[id], is[ih], is[iw], kd, kh, kw, p0, p1, p2, p3, p4, p5, sd, sh, sw);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

class MlasActivationTest : public MlasTestBase
{
public:
    void
    ExecuteShort(
        void
        ) override
    {
        union AliasedValue {
            unsigned u;
            float f;
        };

        // N.B. The test data includes values at the edge of Tanh/Logistic boundaries.
        //    Identity,     Relu,         LeakyRelu,    Tanh,         Logistic,     Clip,
        static const AliasedValue TestData[20][6] = {
            { {0x00000001}, {0x00000001}, {0x00000001}, {0x00000000}, {0x3f000000}, {0x00000001}, }, // positive denormal
            { {0x80000001}, {0x00000000}, {0x80000000}, {0x80000000}, {0x3f000000}, {0x00000000}, }, // negative denormal
            { {0x7ff00002}, {0x7ff00002}, {0x7ff00002}, {0x7ff00002}, {0x7ff00002}, {0x7ff00002}, }, // positive NaN
            { {0xfff00002}, {0xfff00002}, {0xfff00002}, {0xfff00002}, {0xfff00002}, {0xfff00002}, }, // negative NaN
            { {0x00000000}, {0x00000000}, {0x00000000}, {0x00000000}, {0x3f000000}, {0x00000000}, }, // 0.0f
            { {0x80000000}, {0x80000000}, {0x80000000}, {0x80000000}, {0x3f000000}, {0x80000000}, }, // -0.0f
            { {0x3e800000}, {0x3e800000}, {0x3e800000}, {0x3e7acbf5}, {0x3f0feacc}, {0x3e800000}, }, // 0.25f
            { {0xbe800000}, {0x00000000}, {0xbd4ccccd}, {0xbe7acbf5}, {0x3ee02a67}, {0x00000000}, }, // -0.25f
            { {0x40800000}, {0x40800000}, {0x40800000}, {0x3f7fd40a}, {0x3f7b6541}, {0x40800000}, }, // 4.0f
            { {0xc0800000}, {0x00000000}, {0xbf4ccccd}, {0xbf7fd40a}, {0x3c9357e0}, {0x00000000}, }, // -4.0f
            { {0x41200000}, {0x41200000}, {0x41200000}, {0x3f800000}, {0x3f7ffd06}, {0x40c00000}, }, // 10.0f
            { {0xc1200000}, {0x00000000}, {0xc0000000}, {0xbf800000}, {0x383e6000}, {0x00000000}, }, // -10.0f
            { {0xc18866eb}, {0x00000000}, {0xc05a3e45}, {0xbf800000}, {0x33000000}, {0x00000000}, }, // -17.0502529144f
            { {0xc18869bb}, {0x00000000}, {0xc05a42c5}, {0xbf800000}, {0x33c00000}, {0x00000000}, }, // -17.0516262054f
            { {0xc18852a8}, {0x00000000}, {0xc05a1dda}, {0xbf800000}, {0x00000000}, {0x00000000}, }, // -17.0403594971f
            { {0xc18844aa}, {0x00000000}, {0xc05a0777}, {0xbf800000}, {0x00000000}, {0x00000000}, }, // -17.0335273743f
            { {0x418866eb}, {0x418866eb}, {0x418866eb}, {0x3f800000}, {0x3f800000}, {0x40c00000}, }, // +17.0502529144f
            { {0x418869bb}, {0x418869bb}, {0x418869bb}, {0x3f800000}, {0x3f7ffffe}, {0x40c00000}, }, // +17.0516262054f
            { {0x418852a8}, {0x418852a8}, {0x418852a8}, {0x3f800000}, {0x3f800000}, {0x40c00000}, }, // +17.0403594971f
            { {0x418844aa}, {0x418844aa}, {0x418844aa}, {0x3f800000}, {0x3f800000}, {0x40c00000}, }, // +17.0335273743f
        };

        MLAS_ACTIVATION Activation;
        AliasedValue Buffer[_countof(TestData)];

        for (unsigned kind = 0; kind < unsigned(MlasClipActivation); kind++) {

            Activation.ActivationKind = MLAS_ACTIVATION_KIND(kind);

            if (Activation.ActivationKind == MlasLeakyReluActivation) {
                Activation.Parameters.LeakyRelu.alpha = 0.2f;
            } else if (Activation.ActivationKind == MlasClipActivation) {
                Activation.Parameters.Clip.minimum = 0.0f;
                Activation.Parameters.Clip.maximum = 6.0f;
            }

            //
            // Test the vectorized activations.
            //

            for (unsigned i = 0; i < _countof(TestData); i++) {
                Buffer[i].u = TestData[i][0].u;
            }

            MlasActivation(&Activation, &Buffer[0].f, nullptr, 1, _countof(Buffer), _countof(Buffer));

            for (unsigned i = 0; i < _countof(TestData); i++) {
                // Sensitive to comparing positive/negative zero and NaNs.
                if (Buffer[i].u != TestData[i][kind].u && Buffer[i].f != TestData[i][kind].f) {
                    printf("mismatch activation kind=%d i=%d value=%08x expected=%08x\n", (int)kind, (int)i, Buffer[i].u, TestData[i][kind].u);
                }
            }

            //
            // Test the scalar activations.
            //

            for (unsigned i = 0; i < _countof(TestData); i++) {
                Buffer[i].u = TestData[i][0].u;
                MlasActivation(&Activation, &Buffer[i].f, nullptr, 1, 1, 1);
            }

            for (unsigned i = 0; i < _countof(TestData); i++) {
                // Sensitive to comparing positive/negative zero and NaNs.
                if (Buffer[i].u != TestData[i][kind].u && Buffer[i].f != TestData[i][kind].f) {
                    printf("mismatch activation kind=%d i=%d value=%08x expected=%08x\n", (int)kind, (int)i, Buffer[i].u, TestData[i][kind].u);
                }
            }
        }
    }
};

class MlasReorderOutputTest : public MlasTestBase
{
private:
    const size_t BlockSize = MlasNchwcGetBlockSize();

    MatrixGuardBuffer<float> BufferInput;
    MatrixGuardBuffer<float> BufferOutput;
    MatrixGuardBuffer<float> BufferOutput2;
    MatrixGuardBuffer<float> BufferOutputReference;

    void
    Test(
        size_t BatchCount,
        size_t Channels,
        size_t Height,
        size_t Width
        )
    {
        size_t NchwcChannels = (Channels + BlockSize - 1) & ~(BlockSize - 1);

        size_t InputBufferElements = BatchCount * NchwcChannels * Height * Width;
        size_t OutputBufferElements = BatchCount * Channels * Height * Width;

        const float* Input = BufferInput.GetBuffer(InputBufferElements);
        float* Output = BufferOutput.GetBuffer(OutputBufferElements);
        float* OutputReference = BufferOutputReference.GetBuffer(OutputBufferElements);

        int64_t NchwOutputShape[] = { int64_t(BatchCount), int64_t(Channels), int64_t(Height), int64_t(Width) };

        std::fill_n(Output, OutputBufferElements, -0.5f);
        std::fill_n(OutputReference, OutputBufferElements, -0.5f);

        MlasReorderOutputNchw(NchwOutputShape, Input, Output);
        ReferenceReorderOutput(BatchCount, Channels, Height, Width, Input, OutputReference, false);

        if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
            printf("mismatch ReorderOutputNchw: batch=%zd channels=%zd height=%zd width=%zd\n",
                BatchCount, Channels, Height, Width);
        }

        int64_t NhwcOutputShape[] = { int64_t(BatchCount), int64_t(Height), int64_t(Width), int64_t(Channels) };

        std::fill_n(Output, OutputBufferElements, -0.5f);
        std::fill_n(OutputReference, OutputBufferElements, -0.5f);

        MlasReorderOutputNhwc(NhwcOutputShape, Input, Output);
        ReferenceReorderOutput(BatchCount, Channels, Height, Width, Input, OutputReference, true);

        if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
            printf("mismatch ReorderOutputNhwc: batch=%zd channels=%zd height=%zd width=%zd\n",
                BatchCount, Channels, Height, Width);
        }
    }

    void
    ReferenceReorderOutput(
        size_t BatchCount,
        size_t Channels,
        size_t Height,
        size_t Width,
        const float* Input,
        float* Output,
        bool NhwcFormat
        )
    {
        size_t NchwcChannels = (Channels + (BlockSize - 1)) & ~(BlockSize - 1);
        size_t SpatialSize = Height * Width;

        size_t ChannelStride = NhwcFormat ? 1 : SpatialSize;
        size_t SpatialStride = NhwcFormat ? Channels : 1;

        for (size_t n = 0; n < BatchCount; n++) {

            for (size_t c = 0; c < Channels; c++) {

                const float* input = Input + ((c & ~(BlockSize - 1)) * SpatialSize) + (c & (BlockSize - 1));
                float* output = Output + (c * ChannelStride);

                for (size_t hw = 0; hw < SpatialSize; hw++) {
                    output[hw * SpatialStride] = input[hw * BlockSize];
                }
            }

            Input += NchwcChannels * SpatialSize;
            Output += Channels * SpatialSize;
        }
    }

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (size_t c = 1; c < 48; c++) {
            Test(1, c, 112, 112);
            Test(4, c, 15, 21);
            Test(16, c, 11, 11);
        }
    }
};

class MlasSoftmaxTest : public MlasTestBase
{
private:
    MatrixGuardBuffer<float> BufferInput;
    MatrixGuardBuffer<float> BufferOutput;
    MatrixGuardBuffer<float> BufferOutputReference;

    void
    Test(
        size_t N,
        size_t D,
        float MinimumValue,
        float MaximumValue
        )
    {
        float* Input = BufferInput.GetBuffer(N * D);
        float* Output = BufferOutput.GetBuffer(N * D);
        float* OutputReference = BufferOutputReference.GetBuffer(N * D);

        std::default_random_engine generator(static_cast<unsigned>(N * D));
        std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

        for (size_t nd = 0; nd < N * D; nd++) {
            Input[nd] = distribution(generator);
        }

        Test(Input, Output, OutputReference, N, D, false);
        Test(Input, Output, OutputReference, N, D, true);
    }

    void
    Test(
        const float* Input,
        float* Output,
        float* OutputReference,
        size_t N,
        size_t D,
        bool LogSoftmax
        )
    {
        MlasComputeSoftmax(Input, Output, N, D, LogSoftmax, threadpool);
        ReferenceSoftmax(Input, OutputReference, N, D, LogSoftmax);

        constexpr float AbsoluteTolerance = 1e-6f;
        constexpr float RelativeTolerance = 1e-6f;

        for (size_t nd = 0; nd < N * D; nd++) {
            float diff = std::fabs(Output[nd] - OutputReference[nd]);
            if (diff > AbsoluteTolerance && diff > std::fabs(OutputReference[nd]) * RelativeTolerance) {
                printf("softmax(%d) difference: %u/%u %.8f %.8f\n", int32_t(LogSoftmax), unsigned(N), unsigned(D), Output[nd], OutputReference[nd]);
            }
        }
    }

    void
    ReferenceSoftmax(
        const float* Input,
        float* Output,
        size_t N,
        size_t D,
        bool LogSoftmax
        )
    {
        for (size_t n = 0; n < N; n++) {

            float MaximumValue = std::numeric_limits<float>::lowest();

            for (size_t d = 0; d < D; d++) {
                MaximumValue = (std::max)(MaximumValue, Input[d]);
            }

            double Sum = 0.0;

            for (size_t d = 0; d < D; d++) {
                double e = std::exp(double(Input[d]) - double(MaximumValue));
                Sum += e;
                Output[d] = float(e);
            }

            if (LogSoftmax) {

                float Scale = float(std::log(Sum));

                for (size_t d = 0; d < D; d++) {
                    Output[d] = Input[d] - MaximumValue - Scale;
                }

            } else {

                float Scale = float(Sum);

                for (size_t d = 0; d < D; d++) {
                    Output[d] /= Scale;
                }
            }

            Input += D;
            Output += D;
        }
    }

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (size_t d = 1; d < 128; d++) {
            Test(1, d, -10.f, 10.f);
        }

        Test(3, 128, 20.f, 30.f);
        Test(63, 95, -150.f, 190.f);
        Test(16, 211, 20.f, 30.f);
    }
};

class MlasComputeExpTest : public MlasTestBase
{
private:
    MatrixGuardBuffer<float> BufferInput;
    MatrixGuardBuffer<float> BufferOutput;
    MatrixGuardBuffer<float> BufferOutputReference;

    void
    Test(
        size_t N,
        float MinimumValue,
        float MaximumValue
        )
    {
        float* Input = BufferInput.GetBuffer(N);
        float* Output = BufferOutput.GetBuffer(N);
        float* OutputReference = BufferOutputReference.GetBuffer(N);

        std::default_random_engine generator(static_cast<unsigned>(N));
        std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

        for (size_t n = 0; n < N; n++) {
            Input[n] = distribution(generator);
        }

        for (size_t n = 0; n < N; n++) {
            OutputReference[n] = std::exp(Input[n]);
        }

        MlasComputeExp(Input, Output, N);

        constexpr float AbsoluteTolerance = 1e-6f;
        constexpr float RelativeTolerance = 1e-6f;

        for (size_t n = 0; n < N; n++) {
            float diff = std::fabs(Output[n] - OutputReference[n]);
            if (diff > AbsoluteTolerance && diff > std::fabs(OutputReference[n]) * RelativeTolerance) {
                printf("exp difference: %u %.8f %.8f\n", unsigned(N), Output[n], OutputReference[n]);
            }
        }
    }

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (size_t n = 1; n < 128; n++) {
            Test(n, -10.f, 10.f);
        }
    }
};

class MlasQLinearBinaryOpTest : public MlasTestBase
{
public:
    typedef void (MLASCALL *QLinearBinaryOpS8)(
                const int8_t* InputA, float ScaleA, int32_t ZeroPointA,
                const int8_t* InputB, float ScaleB, int32_t ZeroPointB,
                float ScaleC, int32_t ZeroPointC, int8_t* OutputC,
                size_t N, bool IsScalarB);
    typedef void (MLASCALL *QLinearBinaryOpU8)(
                const uint8_t* InputA, float ScaleA, int32_t ZeroPointA,
                const uint8_t* InputB, float ScaleB, int32_t ZeroPointB,
                float ScaleC, int32_t ZeroPointC, uint8_t* OutputC,
                size_t N, bool IsScalarB);

private:
    std::function<float(float, float)> ScalarOp;
    std::string ScalarOpName;
    QLinearBinaryOpS8 QLinearS8Op;
    QLinearBinaryOpU8 QLinearU8Op;
    MatrixGuardBuffer<uint8_t> BufferInputA;
    MatrixGuardBuffer<uint8_t> BufferInputB;
    MatrixGuardBuffer<uint8_t> BufferOutput;
    MatrixGuardBuffer<uint8_t> BufferOutputReference;

    template <typename T>
    T
    QLinearBinaryScalar(
        T a,
        float ScaleA,
        int32_t ZeroPointA,
        T b,
        float ScaleB,
        int32_t ZeroPointB,
        float ScaleC,
        int32_t ZeroPointC
        )
    {
        constexpr int qmax = std::numeric_limits<T>::max();
        constexpr int qmin = std::numeric_limits<T>::min();

        float ValueA = ScaleA * (static_cast<int>(a) - ZeroPointA);
        float ValueB = ScaleB * (static_cast<int>(b) - ZeroPointB);
        float ValueC = std::nearbyintf(ScalarOp(ValueA, ValueB) / ScaleC) + ZeroPointC;
        int qc = static_cast<int>(ValueC);
        qc = std::min(qc, qmax);
        qc = std::max(qc, qmin);
        return static_cast<T>(qc);
    }

    template <typename T>
    void
    Test(
        void (MLASCALL *QLinearBinaryOp)(
                const T* InputA, float ScaleA, int32_t ZeroPointA,
                const T* InputB, float ScaleB, int32_t ZeroPointB,
                float ScaleC, int32_t ZeroPointC, T* OutputC,
                size_t N, bool IsScalarB),
        size_t N,
        bool IsScalarB,
        float ScaleA,
        int32_t ZeroPointA,
        float ScaleB,
        int32_t ZeroPointB,
        float ScaleC,
        int32_t ZeroPointC
        )
    {
        T* InputA = (T*)BufferInputA.GetBuffer(N);
        T* InputB = (T*)BufferInputB.GetBuffer(IsScalarB ? 1 : N);
        T* OutputC = (T*)BufferOutput.GetBuffer(N);
        T* OutputReference = (T*)BufferOutputReference.GetBuffer(N);

        constexpr int MinimumValue = (int)std::numeric_limits<T>::min();
        constexpr int MaximumValue = (int)std::numeric_limits<T>::max();
        std::default_random_engine generator(static_cast<unsigned>(N));
        std::uniform_int_distribution<int> distribution(MinimumValue, MaximumValue);

        if (IsScalarB) {
            InputB[0] = static_cast<T>(distribution(generator));
        }
        for (size_t n = 0; n < N; n++) {
            InputA[n] = static_cast<T>(distribution(generator));
            if (!IsScalarB) {
                InputB[n] = static_cast<T>(distribution(generator));
            }
            OutputReference[n] = QLinearBinaryScalar(InputA[n], ScaleA, ZeroPointA, InputB[IsScalarB ? 0 : n], ScaleB, ZeroPointB, ScaleC, ZeroPointC);
        }

        QLinearBinaryOp(InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N, IsScalarB);

        for (size_t n = 0; n < N; n++) {
            int diff = (int)OutputC[n] - (int)OutputReference[n];
            if (diff < -1 || diff > 1) {
                printf("Test IsScalarB=%d difference @%u of %u, %d(%f,%d) %s %d(%f,%d) => %d(%f,%d) (expecting %d)\n",
                        int(IsScalarB), static_cast<unsigned>(n), static_cast<unsigned>(N),
                        static_cast<int>(InputA[n]), ScaleA, ZeroPointA,
                        ScalarOpName.c_str(),
                        static_cast<int>(InputB[IsScalarB ? 0 : n]), ScaleB, ZeroPointB,
                        static_cast<int>(OutputC[n]), ScaleC, ZeroPointC,
                        static_cast<int>(OutputReference[n]));
            }
        }
    }

public:
    explicit MlasQLinearBinaryOpTest(
        std::function<float(float, float)> P_ScalarOp,
        const std::string& P_ScalarOpName,
        QLinearBinaryOpS8 P_QLinearS8Op,
        QLinearBinaryOpU8 P_QLinearU8Op
        )
        : ScalarOp(P_ScalarOp),
          ScalarOpName(P_ScalarOpName),
          QLinearS8Op(P_QLinearS8Op),
          QLinearU8Op(P_QLinearU8Op)
    {
    }

    void
    ExecuteShort(
        void
        ) override
    {
        static const uint8_t zero_points[] = { 0, 18, 75, 128, 157, 231, 255 };
        static const float c_scales[] = { 18.0f, 90.0f };

        const int8_t* s_zero_points = (const int8_t*)(&zero_points[0]);
        for (size_t a = 0; a < _countof(zero_points); a++) {
            for (size_t b = 0; b < _countof(zero_points); b++) {
                for (size_t c = 0; c < _countof(zero_points); c++) {
                    for (size_t s = 0; s < _countof(c_scales); s++) {
                        for (size_t n = 1; n < 128; n++) {
                            // u8, vector + vector
                            Test<uint8_t>(QLinearU8Op, n, false, 10.f, zero_points[a], 10.f, zero_points[b], c_scales[s], zero_points[c]);

                            // u8, vector + scalar
                            Test<uint8_t>(QLinearU8Op, n, true, 10.f, zero_points[a], 10.f, zero_points[b], c_scales[s], zero_points[c]);

                            // s8, vector + vector
                            Test<int8_t>(QLinearS8Op, n, false, 10.f, s_zero_points[a], 10.f, s_zero_points[b], c_scales[s], s_zero_points[c]);

                            // s8, vector + scalar
                            Test<int8_t>(QLinearS8Op, n, true, 10.f, s_zero_points[a], 10.f, s_zero_points[b], c_scales[s], s_zero_points[c]);
                        }
                    }
                }
            }
        }
    }
};

class MlasQLinearGlobalAveragePoolU8Test : public MlasTestBase
{
private:
    MatrixGuardBuffer<uint8_t> BufferInput;
    MatrixGuardBuffer<uint8_t> BufferOutput;
    MatrixGuardBuffer<uint8_t> BufferOutputReference;

    static
    void
    CalculateGlobalAvgPoolU8(
        const uint8_t* x, int64_t batch, int64_t channel, int64_t hw, bool channel_last,
        uint8_t* y, int32_t x_zero_point, float x_scale, int32_t y_zero_point, float y_scale
        )
    {
        int32_t bias = -x_zero_point * static_cast<int32_t>(hw);
        int64_t stride_image = channel_last ? channel : 1;
        int64_t stride_channel = channel_last ? 1 : hw;

        for (int64_t b = 0; b < batch; ++b) {
            const uint8_t* bx = x + b * hw * channel;
            uint8_t* by = y + b * channel;
            for (int64_t c = 0; c < channel; ++c) {
                const uint8_t* ix = bx + c * stride_channel;
                int32_t sum = 0;
                for (int64_t i = 0; i < hw; ++i) {
                    sum += static_cast<int32_t>(*ix);
                    ix += stride_image;
                }
                sum += bias;
                int32_t r = static_cast<int32_t>(std::nearbyintf(x_scale * sum / static_cast<float>(hw) / y_scale));
                r += y_zero_point;
                r = std::min(255, r);
                r = std::max(0, r);
                by[c] = static_cast<uint8_t>(r);
            }
        }
    }

    static
    void
    CompareResultWithGold(
        size_t Batch, size_t Channel,
        uint8_t* Output, uint8_t* OutputReference, std::string& info
        )
    {
        size_t n = 0;
        for (size_t b =0; b < Batch; ++b) {
            for (size_t c = 0; c < Channel; c++) {
                int diff = abs((int)Output[n] - (int)OutputReference[n]);
                if (diff > 1) {
                    printf("Diff got %d @[%d,%d], Test:%s\n", diff, (int)b, (int)c, info.c_str());
                }
            }
        }
    }

    static
    std::string
    GetTestInfo(
        bool channel_last,
        size_t Batch,
        size_t Stride,
        size_t Channel,
        size_t ImageSize,
        float InputScale,
        uint8_t InputZeroPoint,
        float OutputScale,
        uint8_t OutputZeroPoint
        )
    {
        std::stringstream ss;
        ss << (channel_last ? "Nhwc_" : "Nchw_");
        ss << Batch << "x [C=" << Stride << "-" << Channel << "] x" << ImageSize << "-";
        ss << "(" << (int)InputZeroPoint << "," << InputScale << "," << (int)OutputZeroPoint << "," << OutputScale << ")";
        return ss.str();
    }

    void
    Test(
        bool channel_last,
        size_t Batch,
        size_t Stride,
        size_t Channel,
        size_t ImageSize,
        float InputScale,
        uint8_t InputZeroPoint,
        float OutputScale,
        uint8_t OutputZeroPoint,
        int32_t UnalignedOffset = 0
        )
    {
        size_t N = Batch * Stride * ImageSize;
        size_t ResultLen = Batch * Stride;
        uint8_t* Input = BufferInput.GetBuffer(N);
        uint8_t* Output = BufferOutput.GetBuffer(ResultLen);
        uint8_t* Gold = BufferOutputReference.GetBuffer(ResultLen);
        std::string test_info = GetTestInfo(
            channel_last, Batch, Stride, Channel, ImageSize,
            InputScale, InputZeroPoint, OutputScale, OutputZeroPoint);

        std::default_random_engine generator(static_cast<unsigned>(N));
        std::uniform_int_distribution<int> distribution(0, 255);
        for (size_t n = 0; n < N; n++) {
            Input[n] = static_cast<uint8_t>(distribution(generator));
        }
        CalculateGlobalAvgPoolU8(
            Input, Batch, Stride, ImageSize, channel_last,
            Gold, InputZeroPoint, InputScale, OutputZeroPoint, OutputScale);

        if (!channel_last) {
          std::vector<int32_t> acc(MlasQLinearSafePaddingElementCount(sizeof(int32_t), ResultLen + UnalignedOffset));
          MlasQLinearGlobalAveragePoolNchw(
              Input, InputScale, InputZeroPoint, Output,
              OutputScale, OutputZeroPoint, ResultLen, ImageSize, acc.data() + UnalignedOffset);
        } else {
          std::vector<int32_t> acc(MlasQLinearSafePaddingElementCount(sizeof(int32_t), Channel + UnalignedOffset));
          std::vector<uint8_t> zero(MlasQLinearSafePaddingElementCount(sizeof(uint8_t), Channel + UnalignedOffset));
          if (Stride == Channel) {
            MlasQLinearGlobalAveragePoolNhwc(
                Input, InputScale, InputZeroPoint, Output,
                OutputScale, OutputZeroPoint, Batch, ImageSize, Stride, Channel,
                acc.data() + UnalignedOffset, zero.data() + UnalignedOffset);
            } else {
                for (size_t tc = 0; tc < Stride; tc += Channel) {
                    size_t cg = ((tc + Channel <= Stride) ? Channel : (Stride - tc));
                    MlasQLinearGlobalAveragePoolNhwc(
                        Input + tc, InputScale, InputZeroPoint, Output + tc,
                        OutputScale, OutputZeroPoint, Batch, ImageSize, Stride, cg,
                        acc.data() + UnalignedOffset, zero.data() + UnalignedOffset);
                }
            }
        }

        CompareResultWithGold(Batch, Channel, Output, Gold, test_info);
    }

 public:
    void
    ExecuteShort(
        void
        ) override
    {
        static const uint8_t zero_points[] = {0, 18, 128, 231, 255};
        static const float scales[] = {18.0f, 90.0f};
        static const size_t Batch[] = {1, 3};
        static const size_t Stride[] = {7, 8, 63, 256 };
        static const size_t ImageSize[] = {7, 8, 64};
        static int unalign_offset = 0;

        for (int channel_last = 0; channel_last <= 1; ++channel_last) {
            for (size_t b = 0; b < _countof(Batch); b++) {
                for (size_t xzp = 0; xzp < _countof(zero_points); xzp++) {
                    for (size_t yzp = 0; yzp < _countof(zero_points); yzp++) {
                        for (size_t xs = 0; xs < _countof(scales); ++xs) {
                            for (size_t ys = 0; ys < _countof(scales); ++ys) {
                                for (size_t i = 0; i < _countof(ImageSize); i++) {
                                    for (size_t s = 0; s < _countof(Stride); s++) {
                                        Test(channel_last, Batch[b], Stride[s], Stride[s], ImageSize[i],
                                             scales[xs], zero_points[xzp], scales[ys], zero_points[yzp], unalign_offset);
                                        if (channel_last == 1 && Stride[s] > 32) {
                                            Test(channel_last, Batch[b], Stride[s], 32, ImageSize[i],
                                                 scales[xs], zero_points[xzp], scales[ys], zero_points[yzp], unalign_offset);
                                        }
                                        unalign_offset = (unalign_offset + 1) & 3;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

class MlasFindMinMaxElementsTest : public MlasTestBase
{
private:
    MatrixGuardBuffer<float> BufferInput;

    void
    Test(
        size_t N,
        float MinimumValue,
        float MaximumValue
        )
    {
        float* Input = BufferInput.GetBuffer(N);

        std::default_random_engine generator(static_cast<unsigned>(N));
        std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

        for (size_t n = 0; n < N; n++) {
            Input[n] = distribution(generator);
        }

        auto min_max_pair = std::minmax_element(Input, Input + N);
        float min_ref = *min_max_pair.first;
        float max_ref = *min_max_pair.second;

        float min, max;
        MlasFindMinMaxElement(Input, &min, &max, N);

        constexpr float epsilon = 1e-6f;

        float diff_min = std::fabs(min - min_ref);
        if (diff_min > epsilon) {
            printf("minimum difference: %.8f %.8f\n", min, min_ref);
        }

        float diff_max = std::fabs(max - max_ref);
        if (diff_max > epsilon) {
            printf("maximum difference: %.8f %.8f\n", max, max_ref);
        }
    }

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (size_t n = 1; n < 128; n++) {
            Test(n, -10.f, 10.f);
        }
    }
};

class MlasScaleOutputTest : public MlasTestBase
{
private:
    MatrixGuardBuffer<int32_t> BufferInput;
    MatrixGuardBuffer<float> BufferOutput;
    MatrixGuardBuffer<float> BufferOutputRef;
    MatrixGuardBuffer<float> BufferScale;

    void
    Test(
        size_t M,
        size_t N,
        bool PerColumn,
        bool AccumulateMode
        )
    {
        int32_t* Input = BufferInput.GetBuffer(M * N);
        float* Output = BufferOutput.GetBuffer(M * N);
        float* OutputRef = BufferOutputRef.GetBuffer(M * N);
        float* Scale = BufferScale.GetBuffer(PerColumn ? N : 1);

        std::default_random_engine generator(static_cast<unsigned>(M * N));
        std::uniform_real_distribution<float> real_distribution(-1.0f, 1.0f);
        std::uniform_int_distribution<int32_t> int_distribution(std::numeric_limits<int16_t>::min(),
                                                                std::numeric_limits<int16_t>::max());

        for (size_t s = 0; s < M * N; s++) {

            Input[s] = int_distribution(generator);
            Output[s] = OutputRef[s] = real_distribution(generator);
        }

        for (size_t s = 0; s < (PerColumn ? N : 1); s++) {
            Scale[s] = real_distribution(generator);
        }

        // Compute Reference Value
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {

                float current_scale = PerColumn ? Scale[n] : Scale[0];
                if (AccumulateMode) {
                    OutputRef[m * N + n] += Input[m * N + n] * current_scale;
                } else {
                    OutputRef[m * N + n] = Input[m * N + n] * current_scale;
                }
            }
        }

        // Compute Output with MLAS
        MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR OutputProcessor(Output, N, Scale, nullptr,
                                                               AccumulateMode ? MLAS_QGEMM_OUTPUT_MODE::AccumulateMode : MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
                                                               PerColumn ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);
        OutputProcessor.Process(Input, 0, 0, M, N, N);

        constexpr float epsilon = 1e-6f;

        for (size_t n = 0; n < M * N; n++) {

            float diff = std::fabs(Output[n] - OutputRef[n]);
            if (diff > epsilon) {
                printf("MlasScaleOutputTest: Output[%zu][%zu]:%.8f, OutputRef[%zu][%zu]:%.8f, for case M=%zu, N=%zu\n",
                       n / N, n % N, Output[n], n / N, n % N, OutputRef[n], M, N);
            }
        }
    }

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (size_t m = 1; m < 18; m++) {

            for (size_t n = 1; n < 18; n++) {

                Test(m, n, true, true);
                Test(m, n, true, false);
                Test(m, n, false, true);
                Test(m, n, false, false);
            }
        }
    }
};

template<typename OutputType>
class MlasQuantizeLinearTest : public MlasTestBase
{
private:
    MatrixGuardBuffer<float> BufferInput;
    MatrixGuardBuffer<OutputType> BufferOutput;
    MatrixGuardBuffer<OutputType> BufferOutputReference;

    void
    GenerateReference(
        const float* Input,
        OutputType* OutputReference,
        size_t N,
        float Scale,
        OutputType ZeroPoint
        )
    {
        for (size_t n = 0; n < N; n++) {
            float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
            FloatValue = std::max(FloatValue, float(std::numeric_limits<OutputType>::min()));
            FloatValue = std::min(FloatValue, float(std::numeric_limits<OutputType>::max()));
            OutputReference[n] = (OutputType)FloatValue;
        }
    }

    void
    Test(
        size_t N
        )
    {
        float* Input = BufferInput.GetBuffer(N);
        OutputType* Output = BufferOutput.GetBuffer(N);
        OutputType* OutputReference = BufferOutputReference.GetBuffer(N);

        std::default_random_engine generator(static_cast<unsigned>(N));

        std::uniform_real_distribution<float> min_gen(-10.f, -10e-3f);
        float MinimumValue = min_gen(generator);

        std::uniform_real_distribution<float> max_gen(10e-3f, 10.f);
        float MaximumValue = max_gen(generator);

        float Scale = (MaximumValue - MinimumValue) / 512.f;

        std::uniform_int_distribution<int32_t> zp_distribution(std::numeric_limits<OutputType>::min(), std::numeric_limits<OutputType>::max());
        OutputType ZeroPoint = static_cast<OutputType>(zp_distribution(generator));

        std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);
        for (size_t n = 0; n < N; n++) {
            Input[n] = distribution(generator);
        }

        GenerateReference(Input, OutputReference, N, Scale, ZeroPoint);
        MlasQuantizeLinear(Input, Output, N, Scale, ZeroPoint);

        for (size_t n = 0; n < N; n++) {
            if (Output[n] != OutputReference[n]) {
                printf("exp difference: size=%u, index=%u, output=%d, expected=%d\n", unsigned(N), unsigned(n), int(Output[n]), int(OutputReference[n]));
            }
        }
    }

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (size_t n = 1; n <= 512; n++) {
            Test(n);
        }
    }
};

template<typename ElementType>
class MlasTransposeTest : public MlasTestBase
{
private:
    MatrixGuardBuffer<ElementType> BufferInput;
    MatrixGuardBuffer<ElementType> BufferOutput;
    MatrixGuardBuffer<ElementType> BufferOutputReference;

    void
    Test(
        size_t M,
        size_t N
        )
    {
        ElementType* Input = BufferInput.GetBuffer(M * N);
        ElementType* Output = BufferOutput.GetBuffer(M * N);
        ElementType* OutputReference = BufferOutputReference.GetBuffer(M * N);

        MlasTranspose(Input, Output, M, N);
        ReferenceTranspose(Input, OutputReference, M, N);

        if (memcmp(Output, OutputReference, M * N * sizeof(ElementType)) != 0) {
            printf("mismatch: %zd,%zd (element size %zd)\n", M, N, sizeof(ElementType));
        }
    }

    void
    ReferenceTranspose(
        const ElementType* Input,
        ElementType* Output,
        size_t M,
        size_t N
        )
    {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                Output[n * M + m] = Input[m * N + n];
            }
        }
    }

public:
    void
    ExecuteShort(
        void
        ) override
    {
        for (size_t m = 1; m <= 32; m++) {
            for (size_t n = 1; n <= 32; n++) {
                Test(m, n);
            }
        }
    }
};

void
RunThreadedTests(
    void
    )
{
    printf("SGEMM tests.\n");
    onnxruntime::make_unique<MlasFgemmTest<float, false>>()->ExecuteShort();
    printf("SGEMM packed tests.\n");
    onnxruntime::make_unique<MlasFgemmTest<float, true>>()->ExecuteShort();
#ifdef MLAS_SUPPORTS_GEMM_DOUBLE
    printf("DGEMM tests.\n");
    onnxruntime::make_unique<MlasFgemmTest<double, false>>()->ExecuteShort();
#endif

    printf("QGEMM U8S8=int32_t tests.\n");
    onnxruntime::make_unique<MlasQgemmU8X8Test<int8_t, int32_t, false>>()->ExecuteShort();
    printf("QGEMM U8S8=float tests.\n");
    onnxruntime::make_unique<MlasQgemmU8X8Test<int8_t, float, false>>()->ExecuteShort();
    printf("QGEMM U8U8=int32_t tests.\n");
    onnxruntime::make_unique<MlasQgemmU8X8Test<uint8_t, int32_t, false>>()->ExecuteShort();
    printf("QGEMM U8U8=float tests.\n");
    onnxruntime::make_unique<MlasQgemmU8X8Test<uint8_t, float, false>>()->ExecuteShort();

    if (MlasGemmPackBSize(128, 128, true) > 0) {
        printf("QGEMM U8S8=int32_t packed tests.\n");
        onnxruntime::make_unique<MlasQgemmU8X8Test<int8_t, int32_t, true>>()->ExecuteShort();
        printf("QGEMM U8S8=float packed tests.\n");
        onnxruntime::make_unique<MlasQgemmU8X8Test<int8_t, float, true>>()->ExecuteShort();
    }
    if (MlasGemmPackBSize(128, 128, false) > 0) {
        printf("QGEMM U8U8=int32_t packed tests.\n");
        onnxruntime::make_unique<MlasQgemmU8X8Test<uint8_t, int32_t, true>>()->ExecuteShort();
        printf("QGEMM U8U8=float packed tests.\n");
        onnxruntime::make_unique<MlasQgemmU8X8Test<uint8_t, float, true>>()->ExecuteShort();
    }

    printf("Conv2D tests.\n");
    onnxruntime::make_unique<MlasConv2DTest>()->ExecuteShort();
    if (MlasNchwcGetBlockSize() > 1) {
        onnxruntime::make_unique<MlasNchwcConv2DTest>()->ExecuteShort();
    }

    printf("Pool2D tests.\n");
    onnxruntime::make_unique<MlasPool2DTest>()->ExecuteShort();
    if (MlasNchwcGetBlockSize() > 1) {
        onnxruntime::make_unique<MlasNchwcPool2DTest>()->ExecuteShort();
    }

    printf("Pool3D tests.\n");
    onnxruntime::make_unique<MlasPool3DTest>()->ExecuteShort();

    printf("Softmax tests.\n");
    onnxruntime::make_unique<MlasSoftmaxTest>()->ExecuteShort();
}

int
#if defined(_WIN32)
__cdecl
#endif
main(
    void
    )
{
    //
    // Run threaded tests without the thread pool.
    //

    RunThreadedTests();

#if !defined(MLAS_NO_ONNXRUNTIME_THREADPOOL)

    //
    // Run threaded tests using the thread pool.
    //

    threadpool = new onnxruntime::concurrency::ThreadPool(
        &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr, 2, true);

    RunThreadedTests();

    delete threadpool;

#endif

    //
    // Run remaining tests that do not use the thread pool.
    //

    printf("Activation tests.\n");
    onnxruntime::make_unique<MlasActivationTest>()->ExecuteShort();

    printf("Transcendental tests.\n");
    onnxruntime::make_unique<MlasComputeExpTest>()->ExecuteShort();

    printf("MinMaxElements tests.\n");
    onnxruntime::make_unique<MlasFindMinMaxElementsTest>()->ExecuteShort();

    printf("ReorderOutput tests.\n");
    if (MlasNchwcGetBlockSize() > 1) {
        onnxruntime::make_unique<MlasReorderOutputTest>()->ExecuteShort();
    }

    printf("QLinearAdd tests.\n");
    onnxruntime::make_unique<MlasQLinearBinaryOpTest>(
        [](float a, float b) { return a + b; }, "+", MlasQLinearAdd<int8_t>, MlasQLinearAdd<uint8_t>)->ExecuteShort();

    printf("QLinearMul tests.\n");
    onnxruntime::make_unique<MlasQLinearBinaryOpTest>(
        [](float a, float b) { return a * b; }, "*", MlasQLinearMul<int8_t>, MlasQLinearMul<uint8_t>)->ExecuteShort();

    printf("MlasScaleOutput tests.\n");
    onnxruntime::make_unique<MlasScaleOutputTest>()->ExecuteShort();

    printf("MlasGlobalAveragePool tests.\n");
    onnxruntime::make_unique<MlasQLinearGlobalAveragePoolU8Test>()->ExecuteShort();

    printf("MlasQuantizeLinear tests.\n");
    onnxruntime::make_unique<MlasQuantizeLinearTest<int8_t>>()->ExecuteShort();
    onnxruntime::make_unique<MlasQuantizeLinearTest<uint8_t>>()->ExecuteShort();

    printf("Transpose tests.\n");
    onnxruntime::make_unique<MlasTransposeTest<uint8_t>>()->ExecuteShort();
    onnxruntime::make_unique<MlasTransposeTest<uint32_t>>()->ExecuteShort();

    printf("Done.\n");

    return 0;
}
