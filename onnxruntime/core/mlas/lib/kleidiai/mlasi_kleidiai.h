//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../mlasi.h"
#include <limits>
#include <vector>

// Fix to ensure compatibility with MSVC build
#if defined(_MSC_VER)
  #define RESTRICT __restrict
#else
  #define RESTRICT __restrict__
#endif

// Logging macros.
#ifndef KLEIDIAI_DEBUG_LOGGING
#define KLEIDIAI_DEBUG_LOGGING 0
#endif
#ifndef KLEIDIAI_KERNEL_LOGGING
#define KLEIDIAI_KERNEL_LOGGING 0
#endif

#if KLEIDIAI_DEBUG_LOGGING || KLEIDIAI_KERNEL_LOGGING
#include <iostream>
#define KLEIDIAI_LOG(tag, msg) \
    do { \
        std::cout << "[KLEIDIAI " << tag << "]: " << __FILE__ << " : " << __LINE__ << " : " << msg << std::endl; \
    } while(false)
#endif

// General logging. "tag" is expected to qualify the type of message.
#if KLEIDIAI_DEBUG_LOGGING
    // General debug messages.
    #define KLEIDIAI_DEBUG_LOG(msg) KLEIDIAI_LOG("DEBUG", msg)
#else
    #define KLEIDIAI_DEBUG_LOG(msg)
#endif

#if KLEIDIAI_KERNEL_LOGGING
    // Messages specifically written before a call to kai_run.
    // Note: In cases where a kernel is called in multiple threads, for example MlasTrySimpleParallel,
    // the output order can be inconsistient. The solution is to set the intra-node thread size to 1.
    // If using onnxruntime_perf_test this is done with "--x 1".
    #define KLEIDIAI_KERNEL_LOG(kernel_name) KLEIDIAI_LOG("KERNEL", kernel_name)
#else
    #define KLEIDIAI_KERNEL_LOG(msg)
#endif

namespace ArmKleidiAI {

constexpr size_t MaximumRetainedKleidiAIScratchBytes = 8 * 1024 * 1024;

template <typename T>
void MlasShrinkKleidiAIScratchIfTooLarge(std::vector<T>& buffer)
{
    if (buffer.capacity() > MaximumRetainedKleidiAIScratchBytes / sizeof(T)) {
        std::vector<T>().swap(buffer);
    }
}

// By default we should try for SME2 first before falling back to SME.
inline const bool UseSME2 = MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2();
inline const bool UseSME = MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME();
inline const std::string_view vendor_name = MLAS_CPUIDINFO::GetCPUIDInfo().GetCPUVendor();

// Selects the convolution route for Arm® KleidiAI™
enum class ConvRoute {
    NoKleidiAi,    // decline the conv, caller runs unchanged
    IGemm,         // handle the whole conv via SME IGEMM kernel
    SGemmFallback, // decline IGEMM, but still route the per-segment SGEMM slices through MlasGemm
                   // so that the Arm® KleidiAI™ SGEMM backend override can pick them up
};

struct ConvRouteSelection {
    ConvRoute route = ConvRoute::NoKleidiAi;
    size_t effective_kernel_h = 0;
    size_t effective_kernel_w = 0;
};

// Heuristic default for SME IGEMM selection. Work is estimated as
// output_m * effective_k * filter_count; larger workloads use the SGEMM-backed
// fallback to avoid IGEMM packing overhead on large effective-k convolutions.
inline constexpr size_t ConvIgemmMaxWorkDefault = 1'000'000ULL;

inline bool TryComputeDilatedKernelSize(size_t dilation, size_t kernel, size_t* result) {
    if (dilation == 0 || kernel == 0) return false;

    // using formula: dilated_kernel_size = dilation * (kernel - 1) + 1
    size_t scaled_kernel;
    if (MlasMultiplyOverflowsSizeT(kernel - 1, dilation, &scaled_kernel)) return false;

    if (scaled_kernel == SIZE_MAX) return false;

    *result = scaled_kernel + 1;
    return true;
}

inline bool TryComputeConvOutputSize(size_t input, size_t kernel, size_t padding, size_t stride, size_t* result) {

    if (stride == 0) return false;

    size_t double_padding;
    if (MlasMultiplyOverflowsSizeT(padding, 2, &double_padding)) return false;

    if (double_padding > (SIZE_MAX - input)) return false;
    const size_t padded_input = double_padding + input;

    if (padded_input < kernel) return false;

    // using formula: output_size = ((2*padding + input - kernel) / stride) + 1
    const size_t output_minus_one = (padded_input - kernel) / stride;
    if (output_minus_one == SIZE_MAX) return false;
    *result = output_minus_one + 1;
    return true;
}

inline ConvRouteSelection SelectConvRoute(const MLAS_CONV_PARAMETERS* Parameters) {
    if ((Parameters->Dimensions != 2) ||
        (Parameters->BatchCount != 1) ||
        (Parameters->Beta != 0.f) ||
        (Parameters->Padding[0] != Parameters->Padding[1]) ||
        (Parameters->Padding[0] != Parameters->Padding[2]) ||
        (Parameters->Padding[0] != Parameters->Padding[3])) {
        return ConvRouteSelection{};
    }

    size_t effective_kernel_h;
    size_t effective_kernel_w;
    if (!TryComputeDilatedKernelSize(Parameters->DilationShape[0], Parameters->KernelShape[0], &effective_kernel_h) ||
        !TryComputeDilatedKernelSize(Parameters->DilationShape[1], Parameters->KernelShape[1], &effective_kernel_w)) {
        return ConvRouteSelection{};
    }

    size_t output_m;
    size_t output_h_size;
    size_t output_w_size;
    if (!TryComputeConvOutputSize(Parameters->InputShape[0],
                                  effective_kernel_h,
                                  Parameters->Padding[0],
                                  Parameters->StrideShape[0],
                                  &output_h_size) ||
        !TryComputeConvOutputSize(Parameters->InputShape[1],
                                  effective_kernel_w,
                                  Parameters->Padding[1],
                                  Parameters->StrideShape[1],
                                  &output_w_size) ||
        MlasMultiplyOverflowsSizeT(output_h_size, output_w_size, &output_m)) {
        return ConvRouteSelection{};
    }

    if (output_m == 0) {
        return ConvRouteSelection{};
    }

    if (Parameters->KernelShape[0] < 3 || Parameters->KernelShape[1] < 3) {
        return ConvRouteSelection{};
    }

    const auto filter_count = Parameters->FilterCount;
    if (filter_count == 0) {
        return ConvRouteSelection{};
    }

    size_t effective_k;
    if (MlasMultiplyOverflowsSizeT(Parameters->InputChannels, effective_kernel_h, &effective_k) ||
        MlasMultiplyOverflowsSizeT(effective_k, effective_kernel_w, &effective_k)) {
        return ConvRouteSelection{};
    }
    if (effective_k == 0) {
        return ConvRouteSelection{};
    }

    // Currently, the fallback routes assume NCHW layout, so keep valid NHWC convolutions on IGEMM
    if (Parameters->ChannelsLast) {
        return ConvRouteSelection{ConvRoute::IGemm, effective_kernel_h, effective_kernel_w};
    }

    if (filter_count == 1) {
        // Single-output-channel convolutions do not fill the SME IGEMM N tile efficiently,
        // so defer to the default convolution route.
        return ConvRouteSelection{};
    }

    size_t total_work;
    if (MlasMultiplyOverflowsSizeT(output_m, effective_k, &total_work) ||
        MlasMultiplyOverflowsSizeT(total_work, filter_count, &total_work)) {
        // Total work calculation overflow means total work is too large, fall back
        return ConvRouteSelection{ConvRoute::SGemmFallback, effective_kernel_h, effective_kernel_w};
    }

    const size_t conv_igemm_max_work =
        Parameters->BackendKernelSelectorConfig != nullptr &&
        Parameters->BackendKernelSelectorConfig->kleidiai_conv_igemm_max_work != 0
            ? Parameters->BackendKernelSelectorConfig->kleidiai_conv_igemm_max_work
            : ConvIgemmMaxWorkDefault;
    if (total_work > conv_igemm_max_work) {
        return ConvRouteSelection{ConvRoute::SGemmFallback, effective_kernel_h, effective_kernel_w};
    }

    return ConvRouteSelection{ConvRoute::IGemm, effective_kernel_h, effective_kernel_w};
}

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
MlasGemvBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize
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

#if defined(__aarch64__) && defined(__linux__)
size_t
MLASCALL
MlasSBGemmPackBSize(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K
    );

bool
MLASCALL
MlasSBGemmPackB(
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
MlasSBGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SBGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
    );
#endif

size_t
MLASCALL
MlasDynamicQGemmPackBSize(
    size_t N,
    size_t K
);

void
MLASCALL
MlasDynamicQGemmPackB(
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
                bool ChannelsLast,
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

size_t
MLASCALL
MlasConvSymmetricChannelsLast2DFloatPackWSize(
    size_t FilterCount,
    size_t InputChannels,
    const int64_t* KernelShape,
    const int64_t* DilationShape
    );

void
MLASCALL
MlasConvSymmetricChannelsLast2DFloatPackW(
    size_t FilterCount,
    size_t InputChannels,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    size_t GroupCount,
    const float* Filter,
    const float* Bias,
    void* PackedFilter,
    size_t PackedFilterGroupStride,
    MLAS_THREADPOOL* ThreadPool
    );

inline MLAS_CONV_SGEMM_ROUTE
MLASCALL
MlasConvSGemmRoute(const MLAS_CONV_PARAMETERS* Parameters) {
    if (Parameters->BackendKernelSelectorConfig &&
        !Parameters->BackendKernelSelectorConfig->use_kleidiai) {
        return MlasConvSGemmRouteDirect;
    }

    return ArmKleidiAI::SelectConvRoute(Parameters).route == ArmKleidiAI::ConvRoute::SGemmFallback
        ? MlasConvSGemmRouteDispatch
        : MlasConvSGemmRouteDirect;
}

bool
MLASCALL
MlasHalfGemmBatch(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    const MLAS_HALF_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    );

size_t
MLASCALL
MlasHalfGemmKleidiAIPackBSize(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K
    );

// Packs B into the native KleidiAI RHS-packed layout for the supported
// halfgemm configuration only. This differs from the generic MLAS halfgemm
// prepacked-B format produced by MlasHalfGemmPackB and
// MlasHalfGemmConvertPackB, so generic MLAS prepacked weights may need to be
// repacked into this layout before running the KleidiAI halfgemm path.
// Unsupported transpose combinations return false/0 so the caller can fall
// back to the generic MLAS path.
bool
MLASCALL
MlasHalfGemmKleidiAIPackB(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const MLAS_FP16* B,
    size_t ldb,
    void* PackedB
    );

bool
MLASCALL
MlasHalfConvPrepare(MLAS_CONV_PARAMETERS* Parameters,
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
                    bool InputOutputChannelsLast,
                    MLAS_THREADPOOL* ThreadPool,
                    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig);

bool
MLASCALL
MlasHalfConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const MLAS_FP16* Input,
    const MLAS_FP16* Filter,
    bool FilterAndBiasArePacked,
    const MLAS_FP16* Bias,
    MLAS_FP16* WorkingBuffer,
    MLAS_FP16* Output,
    MLAS_THREADPOOL* ThreadPool
    );

size_t
MLASCALL
MlasHalfConvPackWeightsAndBiasSize(
    size_t FilterCount,
    size_t InputChannels,
    const int64_t* KernelShape,
    const int64_t* DilationShape
    );

bool
MLASCALL
MlasHalfConvPackWeightsAndBias(
    size_t FilterCount,
    size_t InputChannels,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const MLAS_FP16* Filter,
    const MLAS_FP16* Bias,
    void* PackedWeightsAndBias,
    MLAS_THREADPOOL* ThreadPool
    );
}
