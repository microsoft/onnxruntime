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
#include <mlas.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#if !defined(_countof)
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif

class MatrixGuardBuffer
{
public:
    MatrixGuardBuffer(size_t Elements, bool ReadOnly)
    {
        Construct(Elements, ReadOnly);
    }

    ~MatrixGuardBuffer(void)
    {
#if defined(_WIN32)
        VirtualFree(_BaseBuffer, 0, MEM_RELEASE);
#else
        munmap(_BaseBuffer, _BaseBufferSize);
#endif
    }

    float* GetBuffer(size_t Elements)
    {
        return _GuardAddress - Elements;
    }

private:
    void Construct(size_t Elements, bool ReadOnly)
    {
        const size_t PageSize = 4096;
        const size_t GuardPadding = 256 * 1024;

        size_t MatrixSize = Elements * sizeof(float);
        size_t AlignedMatrixSize = (MatrixSize + PageSize - 1) & ~(PageSize - 1);

        _BaseBufferSize = AlignedMatrixSize + GuardPadding;

#if defined(_WIN32)
        _BaseBuffer = VirtualAlloc(NULL, _BaseBufferSize, MEM_RESERVE, PAGE_NOACCESS);
        VirtualAlloc(_BaseBuffer, AlignedMatrixSize, MEM_COMMIT, PAGE_READWRITE);
#else
        _BaseBuffer = mmap(0, _BaseBufferSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        mprotect(_BaseBuffer, AlignedMatrixSize, PROT_READ | PROT_WRITE);
#endif

        float* GuardAddress = (float*)((unsigned char*)_BaseBuffer + AlignedMatrixSize);

        const int MinimumFillValue = -23;
        const int MaximumFillValue = 23;

        int FillValue = MinimumFillValue;
        float* FillAddress = (float*)((unsigned char*)GuardAddress - MatrixSize);

        while (FillAddress < GuardAddress) {

            *FillAddress++ = (float)FillValue;

            FillValue++;

            if (FillValue > MaximumFillValue) {
                FillValue = MinimumFillValue;
            }
        }

        if (ReadOnly) {
#if defined(_WIN32)
            DWORD OldProtect;
            VirtualProtect(_BaseBuffer, AlignedMatrixSize, PAGE_READONLY, &OldProtect);
#else
            mprotect(_BaseBuffer, AlignedMatrixSize, PROT_READ);
#endif
        }

        _GuardAddress = GuardAddress;
    }

private:
    void* _BaseBuffer;
    size_t _BaseBufferSize;
    float* _GuardAddress;
};

void
ReferenceSgemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc
    )
{
    if (TransA == CblasNoTrans) {

        if (TransB == CblasNoTrans) {

            for (size_t m = 0; m < M; m++) {

                for (size_t n = 0; n < N; n++) {

                    const float* a = A + (m * lda);
                    const float* b = B + n;
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

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

                    const float* a = A + (m * lda);
                    const float* b = B + (n * ldb);
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

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

                    const float* a = A + m;
                    const float* b = B + n;
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

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

                    const float* a = A + m;
                    const float* b = B + (n * ldb);
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

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

void
TrialSgemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    float* CReference,
    size_t ldc
    )
{
    for (size_t f = 0; f < M * N; f++) {
        C[f] = -0.5f;
        CReference[f] = -0.5f;
    }

    MlasSgemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, nullptr);
    ReferenceSgemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, CReference, ldc);

    for (size_t f = 0; f < M * N; f++) {
        // Sensitive to comparing positive/negative zero.
        if (C[f] != CReference[f]) {
            printf("mismatch TransA=%d, TransB=%d, M=%zd, N=%zd, K=%zd, alpha=%f, beta=%f!\n", TransA, TransB, M, N, K, alpha, beta);
        }
    }
}

void
TrialSgemm(
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    MatrixGuardBuffer& BufferA,
    MatrixGuardBuffer& BufferB,
    float beta,
    MatrixGuardBuffer& BufferC,
    MatrixGuardBuffer& BufferCReference
    )
{
    const float* A = BufferA.GetBuffer(K * M);
    const float* B = BufferB.GetBuffer(N * K);
    float* C = BufferC.GetBuffer(N * M);
    float* CReference = BufferCReference.GetBuffer(N * M);

    TrialSgemm(CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, CReference, N);
    TrialSgemm(CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, CReference, N);
    TrialSgemm(CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, CReference, N);
    TrialSgemm(CblasTrans, CblasTrans, M, N, K, alpha, A, M, B, K, beta, C, CReference, N);
}

void
ExecuteSgemmTests(
    void
    )
{
    constexpr size_t MaximumDimension = 320;

    MatrixGuardBuffer BufferA(MaximumDimension * MaximumDimension, true);
    MatrixGuardBuffer BufferB(MaximumDimension * MaximumDimension, true);
    MatrixGuardBuffer BufferC(MaximumDimension * MaximumDimension, false);
    MatrixGuardBuffer BufferCReference(MaximumDimension * MaximumDimension, false);

    // Trial balloons.
    for (size_t b = 1; b < 16; b++) {
        TrialSgemm(b, b, b, 1.0f, BufferA, BufferB, 0.0f, BufferC, BufferCReference);
    }
    for (size_t b = 16; b <= 256; b <<= 1) {
        TrialSgemm(b, b, b, 1.0f, BufferA, BufferB, 0.0f, BufferC, BufferCReference);
    }
    for (size_t b = 256; b < 320; b += 32) {
        TrialSgemm(b, b, b, 1.0f, BufferA, BufferB, 0.0f, BufferC, BufferCReference);
    }

    static const float multipliers[] = { 0.0f, -0.0f, 0.25f, -0.5f, 1.0f, -1.0f };

    for (size_t N = 1; N < 128; N++) {
        for (size_t K = 1; K < 128; K++) {
            for (size_t a = 0; a < _countof(multipliers); a++) {
                for (size_t b = 0; b < _countof(multipliers); b++) {
                    TrialSgemm(1, N, K, multipliers[a], BufferA, BufferB, multipliers[b], BufferC, BufferCReference);
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

                        TrialSgemm(M, N, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 1, N, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M, N + 1, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 1, N + 1, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 3, N + 2, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 4, N, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M, N + 4, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 4, N + 4, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 3, N + 7, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 8, N, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M, N + 8, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 12, N + 12, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 13, N, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M, N + 15, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 15, N + 15, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                    }
                }
                printf("a %zd/%zd b %zd/%zd M %zd\n", a, _countof(multipliers), b, _countof(multipliers), M);
            }
        }
    }

    for (size_t M = 1; M < 160; M++) {
        for (size_t N = 1; N < 160; N++) {
            for (size_t K = 1; K < 160; K++) {
                TrialSgemm(M, N, K, 1.0f, BufferA, BufferB, 0.0f, BufferC, BufferCReference);
            }
        }
        printf("M %zd\n", M);
    }

    for (size_t M = 160; M < 320; M += 24) {
        for (size_t N = 112; N < 320; N += 24) {
            for (size_t K = 1; K < 16; K++) {
                TrialSgemm(M, N, K, 1.0f, BufferA, BufferB, 0.0f, BufferC, BufferCReference);
            }
            for (size_t K = 16; K < 160; K += 32) {
                TrialSgemm(M, N, K, 1.0f, BufferA, BufferB, 0.0f, BufferC, BufferCReference);
            }
        }
        printf("M %zd\n", M);
    }
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
    size_t Im2ColBufferElements = OutputSize * K;

    MatrixGuardBuffer BufferIm2Col(Im2ColBufferElements, false);

    for (size_t b = 0; b < BatchCount; b++) {

        const float* filter = Filter;
        const float* bias = Bias;

        for (size_t g = 0; g < GroupCount; g++) {

            //
            // Transform the image using IM2COL and invoke the GEMM.
            //

            float* Im2Col = BufferIm2Col.GetBuffer(Im2ColBufferElements);
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

            MlasSgemm(CblasNoTrans, CblasNoTrans, FilterCount, OutputSize, K, 1.0f,
                filter, K, Im2Col, OutputSize, 0.0f, Output, OutputSize, nullptr);

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

void
TrialConv2D(
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

    int64_t InputShape[] = { int64_t(InputHeight), int64_t(InputWidth) };
    int64_t KernelShape[] = { int64_t(KernelHeight), int64_t(KernelWidth) };
    int64_t DilationShape[] = { int64_t(DilationHeight), int64_t(DilationWidth) };
    int64_t Padding[] = { int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth) };
    int64_t StrideShape[] = { int64_t(StrideHeight), int64_t(StrideWidth) };
    int64_t OutputShape[] = { OutputHeight64, OutputWidth64 };

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

    size_t OutputHeight = size_t(OutputHeight64);
    size_t OutputWidth = size_t(OutputWidth64);

    size_t InputSize = InputHeight * InputWidth;
    size_t KernelSize = KernelHeight * KernelWidth;
    size_t OutputSize = OutputHeight * OutputWidth;

    size_t InputBufferElements = BatchCount * GroupCount * InputChannels * InputSize;
    size_t FilterBufferElements = GroupCount * FilterCount * InputChannels * KernelSize;
    size_t BiasBufferElements = GroupCount * FilterCount;
    size_t OutputBufferElements = BatchCount * GroupCount * FilterCount * OutputSize;

    MatrixGuardBuffer BufferInput(InputBufferElements, true);
    MatrixGuardBuffer BufferFilter(FilterBufferElements, true);
    MatrixGuardBuffer BufferBias(BiasBufferElements, true);
    MatrixGuardBuffer BufferOutput(OutputBufferElements, false);
    MatrixGuardBuffer BufferOutputReference(OutputBufferElements, false);

    const float* Input = BufferInput.GetBuffer(InputBufferElements);
    const float* Filter = BufferFilter.GetBuffer(FilterBufferElements);
    const float* Bias = BufferBias.GetBuffer(BiasBufferElements);
    float* Output = BufferOutput.GetBuffer(OutputBufferElements);
    float* OutputReference = BufferOutputReference.GetBuffer(OutputBufferElements);

    MatrixGuardBuffer BufferWorking(WorkingBufferSize, false);

    MlasConv(&Parameters,
             Input,
             Filter,
             Bias,
             BufferWorking.GetBuffer(WorkingBufferSize),
             Output,
             nullptr);

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

    if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
        printf("mismatch: batch=%zd,group=%zd,input(%zd,%zd,%zd),filter=%zd,kernel(%zd,%zd)!!!\n",
            BatchCount, GroupCount, InputChannels, InputHeight, InputWidth, FilterCount,
            KernelHeight, KernelWidth);
    }
}

void
ExecuteConvTests(
    void
    )
{
    static const unsigned cs[] = { 32, 14, 1 };
    static const unsigned is[] = { 53, 11, 5, 1 };

    for (unsigned i = 1; i < 256; i <<= 1) {
        TrialConv2D(1, 1, 16, i, i, 32, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
        TrialConv2D(1, 1, 16, i, i, 32, 3, 3, 0, 0, 0, 0, 1, 1, 2, 2);
        TrialConv2D(1, 1, 16, i, i, 32, 3, 3, 0, 0, 0, 0, 2, 2, 1, 1);
        TrialConv2D(1, 1, 16, i, i, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
        TrialConv2D(1, 1, 16, i, i, 32, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
        TrialConv2D(1, 1, 16, i, i, 32, i, 1, 0, 0, 0, 0, 1, 1, 1, 1);
        TrialConv2D(1, 1, 16, i, i, 32, 1, i, 0, 0, 0, 0, 1, 1, 1, 1);
    }

    for (unsigned i = 1; i <= 32; i++) {
        TrialConv2D(4, 18, 1, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
        TrialConv2D(4, 18, 1, 32, 89, 48, i, 89, 1, 1, 1, 1, 1, 1, 1, 1);
        TrialConv2D(4, 18, 2, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
    }

    for (unsigned b = 1; b < 64; b++) {
        TrialConv2D(b, 1, 64, 11, 11, 128, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
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
                                                            TrialConv2D(1, 1, cs[ic], is[ih], is[iw], cs[fc], kh, kw, p0, p1, p2, p3, dh, dw, sh, sw);
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

void
TrialPool2D(
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
    int64_t InputShape[] = { int64_t(BatchCount), int64_t(InputChannels), int64_t(InputHeight), int64_t(InputWidth) };
    int64_t KernelShape[] = { int64_t(KernelHeight), int64_t(KernelWidth) };
    int64_t Padding[] = { int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth) };
    int64_t StrideShape[] = { int64_t(StrideHeight), int64_t(StrideWidth) };
    int64_t OutputShape[] = { int64_t(BatchCount), int64_t(InputChannels), 0, 0 };

    OutputShape[2] = (InputShape[2] + Padding[0] + Padding[2] - KernelShape[0]) / StrideShape[0] + 1;
    OutputShape[3] = (InputShape[3] + Padding[1] + Padding[3] - KernelShape[1]) / StrideShape[1] + 1;

    size_t InputBufferElements = size_t(InputShape[0] * InputShape[1] * InputShape[2] * InputShape[3]);
    size_t OutputBufferElements = size_t(OutputShape[0] * OutputShape[1] * OutputShape[2] * OutputShape[3]);

    MatrixGuardBuffer BufferInput(InputBufferElements, true);
    MatrixGuardBuffer BufferOutput(OutputBufferElements, false);
    MatrixGuardBuffer BufferOutputReference(OutputBufferElements, false);

    const float* Input = BufferInput.GetBuffer(InputBufferElements);
    float* Output = BufferOutput.GetBuffer(OutputBufferElements);
    float* OutputReference = BufferOutputReference.GetBuffer(OutputBufferElements);

    MlasPool(MlasMaximumPooling, 2, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output, nullptr);
    ReferenceMaximumPool2D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference);

    if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
        printf("mismatch: maximum input(%zd,%zd,%zd),kernel(%zd,%zd)!!!\n",
            InputChannels, InputHeight, InputWidth, KernelHeight, KernelWidth);
    }

    MlasPool(MlasAveragePoolingExcludePad, 2, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output, nullptr);
    ReferenceAveragePool2D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference, false);

    if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
        printf("mismatch: averageexcpad input(%zd,%zd,%zd),kernel(%zd,%zd)!!!\n",
            InputChannels, InputHeight, InputWidth, KernelHeight, KernelWidth);
    }

    MlasPool(MlasAveragePoolingIncludePad, 2, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output, nullptr);
    ReferenceAveragePool2D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference, true);

    if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
        printf("mismatch: averageincpad input(%zd,%zd,%zd),kernel(%zd,%zd)!!!\n",
            InputChannels, InputHeight, InputWidth, KernelHeight, KernelWidth);
    }
}

void
TrialPool3D(
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
    int64_t InputShape[] = { int64_t(BatchCount), int64_t(InputChannels), int64_t(InputDepth), int64_t(InputHeight), int64_t(InputWidth) };
    int64_t KernelShape[] = { int64_t(KernelDepth), int64_t(KernelHeight), int64_t(KernelWidth) };
    int64_t Padding[] = { int64_t(PaddingLeftDepth), int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightDepth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth) };
    int64_t StrideShape[] = { int64_t(StrideDepth), int64_t(StrideHeight), int64_t(StrideWidth) };
    int64_t OutputShape[] = { int64_t(BatchCount), int64_t(InputChannels), 0, 0, 0 };

    OutputShape[2] = (InputShape[2] + Padding[0] + Padding[3] - KernelShape[0]) / StrideShape[0] + 1;
    OutputShape[3] = (InputShape[3] + Padding[1] + Padding[4] - KernelShape[1]) / StrideShape[1] + 1;
    OutputShape[4] = (InputShape[4] + Padding[2] + Padding[5] - KernelShape[2]) / StrideShape[2] + 1;

    size_t InputBufferElements = size_t(InputShape[0] * InputShape[1] * InputShape[2] * InputShape[3] * InputShape[4]);
    size_t OutputBufferElements = size_t(OutputShape[0] * OutputShape[1] * OutputShape[2] * OutputShape[3] * OutputShape[4]);

    MatrixGuardBuffer BufferInput(InputBufferElements, true);
    MatrixGuardBuffer BufferOutput(OutputBufferElements, false);
    MatrixGuardBuffer BufferOutputReference(OutputBufferElements, false);

    const float* Input = BufferInput.GetBuffer(InputBufferElements);
    float* Output = BufferOutput.GetBuffer(OutputBufferElements);
    float* OutputReference = BufferOutputReference.GetBuffer(OutputBufferElements);

    MlasPool(MlasMaximumPooling, 3, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output, nullptr);
    ReferenceMaximumPool3D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference);

    if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
        printf("mismatch: maximum input(%zd,%zd,%zd,%zd),kernel(%zd,%zd,%zd)!!!\n",
            InputChannels, InputDepth, InputHeight, InputWidth, KernelDepth, KernelHeight, KernelWidth);
    }

    MlasPool(MlasAveragePoolingExcludePad, 3, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output, nullptr);
    ReferenceAveragePool3D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference, false);

    if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
        printf("mismatch: averageexcpad input(%zd,%zd,%zd,%zd),kernel(%zd,%zd,%zd)!!!\n",
            InputChannels, InputDepth, InputHeight, InputWidth, KernelDepth, KernelHeight, KernelWidth);
    }

    MlasPool(MlasAveragePoolingIncludePad, 3, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output, nullptr);
    ReferenceAveragePool3D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference, true);

    if (memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)) != 0) {
        printf("mismatch: averageincpad input(%zd,%zd,%zd,%zd),kernel(%zd,%zd,%zd)!!!\n",
            InputChannels, InputDepth, InputHeight, InputWidth, KernelDepth, KernelHeight, KernelWidth);
    }
}

void
ExecutePool2DTests(
    void
    )
{
    static const unsigned is[] = { 53, 17, 11, 5, 4, 3, 2, 1 };

    for (unsigned i = 1; i < 2058; i++) {
        TrialPool2D(1, 1, 4, i, 2, 4, 0, 2, 0, 1, 1, 1);
    }

    for (unsigned ih = 0; ih < _countof(is); ih++) {
        for (unsigned iw = 0; iw < _countof(is); iw++) {
            fprintf(stderr, "Handling %ux%u\n", is[ih], is[iw]);
            TrialPool2D(1, 1, is[ih], is[iw], is[ih], is[iw], 0, 0, 0, 0, 1, 1);
            TrialPool2D(1, 1, is[ih], is[iw], is[ih], 1, 0, 0, 0, 0, 1, 1);
            TrialPool2D(1, 1, is[ih], is[iw], 1, is[iw], 0, 0, 0, 0, 1, 1);
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
                                            TrialPool2D(2, 3, is[ih], is[iw], kh, kw, p0, p1, p2, p3, sh, sw);
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

void
ExecutePool3DTests(
    void
    )
{
    static const unsigned is[] = { 11, 5, 4, 3, 2, 1 };

    for (unsigned id = 0; id < _countof(is); id++) {
        for (unsigned ih = 0; ih < _countof(is); ih++) {
            for (unsigned iw = 0; iw < _countof(is); iw++) {
                fprintf(stderr, "Handling %ux%ux%u\n", is[id], is[ih], is[iw]);
                TrialPool3D(1, 1, is[id], is[ih], is[iw], is[id], is[ih], is[iw], 0, 0, 0, 0, 0, 0, 1, 1, 1);
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
                                                                TrialPool3D(1, 1, is[id], is[ih], is[iw], kd, kh, kw, p0, p1, p2, p3, p4, p5, sd, sh, sw);
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

#if 0
#if defined(_WIN32)

extern uint32_t MlasMaximumThreadCount;

void
EvaluateThreadingPerformance(
    void
    )
{
    SYSTEM_INFO SystemInfo;

    GetSystemInfo(&SystemInfo);

    const size_t MaximumDimension = 4096;

    MatrixGuardBuffer BufferA(MaximumDimension, true);
    MatrixGuardBuffer BufferB(MaximumDimension, true);
    MatrixGuardBuffer BufferC(MaximumDimension, false);

    for (size_t M = 16; M <= MaximumDimension; M <<= 1) {
        for (size_t N = 16; N <= MaximumDimension; N <<= 1) {
            for (size_t K = 16; K <= MaximumDimension; K <<= 1) {

                const float* A = BufferA.GetBuffer(K * M);
                const float* B = BufferB.GetBuffer(N * K);
                float* C = BufferC.GetBuffer(N * M);

                //
                // Compute the number of iterations to run for at least five seconds.
                //

                MlasMaximumThreadCount = 1;

                ULONG64 NumberIterations = 0;
                DWORD start = GetTickCount();
                DWORD stop;
                do {
                    MlasSgemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N, nullptr);
                    stop = GetTickCount();
                    NumberIterations++;
                } while ((stop - start) <= 5000);

                //
                // Determine the performance for a range of thread counts.
                //

                static DWORD cpus[] = { 1, 2, 4, 8, 12 };
                for (size_t cpu = 0; cpu < _countof(cpus); cpu++) {

                    if (cpus[cpu] <= SystemInfo.dwNumberOfProcessors) {
                        MlasMaximumThreadCount = cpus[cpu];
                    } else {
                        break;
                    }

                    start = GetTickCount();
                    for (size_t iters = 0; iters < NumberIterations; iters++) {
                        MlasSgemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N, nullptr);
                        stop = GetTickCount();
                        if ((stop - start) > 20000) {
                            break;
                        }
                    }

                    printf("%zd,%zd,%zd cpus=%d, iters=%I64d, time=%d %c\n", M, N, K,
                        MlasMaximumThreadCount, NumberIterations, stop - start, ((stop - start) > 5100) ? '!' : '\0');
                }

                fflush(stdout);
            }
        }
    }
}

#endif
#endif

int
#if defined(_WIN32)
__cdecl
#endif
main(
    void
    )
{
//    ExecuteSgemmTests();
    ExecuteConvTests();
//    ExecutePool2DTests();
//    ExecutePool3DTests();
//    EvaluateThreadingPerformance();

    return 0;
}
