/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas.h

Abstract:

    This module contains the public data structures and procedure prototypes
    for the Microsoft Machine Learning algebra subprogram library.

--*/

#pragma once

#include <cstdlib>
#include <cstdint>

//
// Define the calling convention for Windows targets.
//

#if (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)
#define MLASCALL __stdcall
#else
#define MLASCALL
#endif

//
// Basic Linear Algebra Subprograms (BLAS) types.
//

#ifndef CBLAS_ENUM_DEFINED_H
#define CBLAS_ENUM_DEFINED_H
typedef enum { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;
typedef enum { CblasUpper=121, CblasLower=122 } CBLAS_UPLO;
typedef enum { CblasNonUnit=131, CblasUnit=132 } CBLAS_DIAG;
typedef enum { CblasLeft=141, CblasRight=142} CBLAS_SIDE;
#endif

//
// Forward declare the thread pool implementation class.
//
// N.B. Avoid including ONNX Runtime headers here to keep the dependencies for
// standalone MLAS test executables smaller.
//

namespace onnxruntime {
    namespace concurrency {
        class ThreadPool;
    };
};

using MLAS_THREADPOOL = onnxruntime::concurrency::ThreadPool;

//
// Platform routines.
//

size_t
MLASCALL
MlasGetPreferredBufferAlignment(
    void
    );

//
// Activation routines.
//

enum MLAS_ACTIVATION_KIND {
    MlasIdentityActivation,
    MlasReluActivation,
    MlasLeakyReluActivation,
    MlasTanhActivation,
    MlasLogisticActivation,
    MlasClipActivation,
};

struct MLAS_ACTIVATION {
    MLAS_ACTIVATION_KIND ActivationKind;
    union {
        struct {
            float alpha;
        } LeakyRelu;
        struct {
            float minimum;
            float maximum;
        } Clip;
        float Values[2];
    } Parameters;
};

void
MLASCALL
MlasActivation(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc
    );

//
// Matrix/matrix multiply routines.
//

void
MLASCALL
MlasGemm(
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
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    double alpha,
    const double* A,
    size_t lda,
    const double* B,
    size_t ldb,
    double beta,
    double* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasGemm(
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
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasGemm(
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
    const float* Scale,
    const float* Bias,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasGemm(
    size_t M,
    size_t N,
    size_t K,
    const uint8_t* A,
    size_t lda,
    uint8_t offa,
    const void* PackedB,
    uint8_t offb,
    bool BIsSigned,
    int32_t* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasGemm(
    size_t M,
    size_t N,
    size_t K,
    const uint8_t* A,
    size_t lda,
    uint8_t offa,
    const void* PackedB,
    uint8_t offb,
    bool BIsSigned,
    float* C,
    size_t ldc,
    const float* Scale,
    const float* Bias,
    MLAS_THREADPOOL* ThreadPool
    );

//
// Buffer packing routines.
//

size_t
MLASCALL
MlasGemmPackBSize(
    size_t N,
    size_t K,
    bool BIsSigned
    );

void
MLASCALL
MlasGemmPackB(
    size_t N,
    size_t K,
    const uint8_t* B,
    size_t ldb,
    bool BIsSigned,
    void* PackedB
    );

//
// Convolution routines.
//

enum MLAS_CONV_ALGORITHM {
    MlasConvAlgorithmGemmDirect,
    MlasConvAlgorithmExpandThenGemm,
    MlasConvAlgorithmExpandThenGemmSegmented,
};

struct MLAS_CONV_PARAMETERS {
    const MLAS_ACTIVATION* Activation;
    size_t Dimensions;
    size_t BatchCount;
    size_t GroupCount;
    size_t InputChannels;
    size_t InputShape[3];
    size_t KernelShape[3];
    size_t DilationShape[3];
    size_t Padding[6];
    size_t StrideShape[3];
    size_t FilterCount;
    size_t OutputShape[3];
    size_t InputSize;
    size_t OutputSize;
    size_t K;
    MLAS_CONV_ALGORITHM Algorithm;
    int32_t ThreadCount;
    union {
        struct {
            CBLAS_TRANSPOSE TransB;
            size_t ldb;
        } GemmDirect;
        struct {
            size_t ThreadStrideN;
        } ExpandThenGemmSegmented;
    } u;
};

void
MLASCALL
MlasConvPrepare(
    MLAS_CONV_PARAMETERS* Parameters,
    size_t Dimensions,
    size_t BatchCount,
    size_t GroupCount,
    size_t InputChannels,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    size_t FilterCount,
    const MLAS_ACTIVATION* Activation,
    size_t* WorkingBufferSize,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    );

//
// Pooling routines.
//

enum MLAS_POOLING_KIND {
    MlasMaximumPooling,
    MlasAveragePoolingExcludePad,
    MlasAveragePoolingIncludePad,
    MlasPoolingKindCount,
};

void
MLASCALL
MlasPool(
    MLAS_POOLING_KIND PoolingKind,
    size_t Dimensions,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    const float* Input,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    );

//
// Miscellaneous compute routines.
//

void
MLASCALL
MlasComputeErf(
    const float* Input,
    float* Output,
    size_t N
    );

void
MLASCALL
MlasComputeExp(
    const float* Input,
    float* Output,
    size_t N
    );

void
MLASCALL
MlasComputeLogistic(
    const float* Input,
    float* Output,
    size_t N
    );

void
MLASCALL
MlasComputeSoftmax(
    const float* Input,
    float* Output,
    size_t N,
    size_t D,
    bool LogSoftmax,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasComputeTanh(
    const float* Input,
    float* Output,
    size_t N
    );

//
// Half-precision floating-point routines.
//

extern "C"
void
MLASCALL
MlasConvertHalfToFloatBuffer(
    const unsigned short* Source,
    float* Destination,
    size_t Count
    );

//
// Buffer reordering routines.
//

void
MLASCALL
MlasReorderInput(
    const int64_t* InputShape,
    const float* S,
    float* D
    );

void
MLASCALL
MlasReorderOutputNchw(
    const int64_t* OutputShape,
    const float* S,
    float* D
    );

void
MLASCALL
MlasReorderOutputNhwc(
    const int64_t* OutputShape,
    const float* S,
    float* D
    );

void
MLASCALL
MlasReorderFilterOIHWBiBo(
    const int64_t* FilterShape,
    const float* S,
    float* D
    );

void
MLASCALL
MlasReorderFilterOIHWBo(
    const int64_t* FilterShape,
    const float* S,
    float* D
    );

//
// Single precision NCHWc routines.
//

size_t
MLASCALL
MlasNchwcGetBlockSize(
    void
    );

void
MLASCALL
MlasNchwcConv(
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    size_t GroupCount,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* Output,
    const MLAS_ACTIVATION* Activation,
    bool ZeroMode,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasNchwcPool(
    MLAS_POOLING_KIND PoolingKind,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    const float* Input,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    );

void
MLASCALL
MlasNchwcUpsample(
    const int64_t* InputShape,
    const int64_t* Scales,
    const float* Input,
    float* Output
    );

//
// Linear quantization routines.
//

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    );

void
MLASCALL
MlasRequantizeOutput(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    );

void
MLASCALL
MlasFindMinMaxElement(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
    );

//
// InputA is of size N,
// Input B is of size 1 if IsScalarB == true, otherwise it is of size N
//
template<typename DataType>
void
MLASCALL
MlasQLinearAdd(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t N,
    bool IsScalarB
    );
