//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#pragma once

#include "mlasi.h"
#include <iostream>

// Fix to ensure compatibility with MSVC build
#if defined(_MSC_VER)
  #define RESTRICT __restrict
#else
  #define RESTRICT __restrict__
#endif

// Logging macros.
#ifndef KLEIDIAI_DEBUG
#define KLEIDIAI_DEBUG 0
#endif
#ifndef KLEIDIAI_KERNEL
#define KLEIDIAI_KERNEL 0
#endif

// General logging. "tag" is expected to quality the type of message.
#define KLEIDIAI_LOG(tag, msg) std::cout << "[KLEIDIAI " << tag << "]: " << msg << std::endl;

#if KLEIDIAI_DEBUG
    // General debug messages.
    #define KLEIDIAI_DEBUG_LOG(msg) KLEIDIAI_LOG("DEBUG", msg)
#else
    #define KLEIDIAI_DEBUG_LOG(msg)
#endif

#if KLEIDIAI_KERNEL
    // Messages specifically written before a call to kai_run.
    // Note: In cases where a kernel is called in multiple threads, for example MlasTrySimpleParallel,
    // the output order can be inconsistient. The solution is to set the intra-node thread size to 1.
    // If using onnxruntime_perf_test this is done with "--x 1".
    #define KLEIDIAI_KERNEL_LOG(kernel_name) KLEIDIAI_LOG("KERNEL", kernel_name)
#else
    #define KLEIDIAI_KERNEL_LOG(msg)
#endif

namespace ArmKleidiAI {
// By default we should try for SME2 first before falling back to SME.
inline const bool UseSME2 = MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2();

//
// Buffer packing routines.
//

size_t
MLASCALL
MlasGemmPackBSize(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K
    );

bool
MLASCALL
MlasGemmPackB(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB
    );

bool
MLASCALL
MlasGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
    );

size_t
MLASCALL
MlasDynamicQgemmPackBSize(
    size_t N,
    size_t K
);

void
MLASCALL
MlasDynamicQgemmPackB(
    size_t N,
    size_t K,
    const int8_t* B,
    const float* Scales,
    const float* Bias,
    void* PackedB
);

//pack symmetric quantized B and dynamic quantized A
void
MLASCALL
MlasDynamicQGemmBatch(
    const MLAS_GEMM_DYN_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_DYN_QUANT_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool
    );

bool
MLASCALL
MlasConvPrepare(MLAS_CONV_PARAMETERS* Parameters,
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
                float Beta,
                MLAS_THREADPOOL* ThreadPool);

bool
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
}
