#pragma once
#include "qnbitgemm.h"

size_t Q2BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
);

void
SQ2BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /* ComputeType*/,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool
);

size_t
Q2BitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
);

void
GenerateLUT_avx2(
    int32_t group_size,
    int8_t lut,
    const float* b,
    float* scales,
    float* biases,
    int K
);

void
TMACComputeGemm_avx2(
    const void* A,
    const void* a_scales,
    const void* LUT,
    const void* LUT_Scales,
    const void* LUT_Biases,
    void* C,
    int bm,
    int K,
    int M,                
    int N,
    size_t BlkLen
);