/*++

Copyright (c) Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    cast.cpp

Abstract:

    This module implements Half (F16) to Single (F32) precision casting.

--*/
#include "mlasi.h"

void
MLASCALL
MlasConvertHalfToFloatBuffer(
    const MLAS_FP16* Source,
    float* Destination,
    size_t Count
)
{
    if (GetMlasPlatform().CastF16ToF32Kernel == nullptr) {
        for (size_t i = 0; i < Count; ++i) {
            Destination[i] = Source[i].ToFloat();
        }
    } else {
        // If the kernel is available, use it to perform the conversion.
        GetMlasPlatform().CastF16ToF32Kernel(reinterpret_cast<const unsigned short*>(Source), Destination, Count);
    }
}


void
MLASCALL
MlasConvertHalfToFloatBufferInParallel(
    const MLAS_FP16* Source,
    float* Destination,
    size_t Count,
    MLAS_THREADPOOL* ThreadPool
)
{
#if defined(BUILD_MLAS_NO_ONNXRUNTIME)
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    // If the ThreadPool is not available, use the single-threaded version.
    MlasConvertHalfToFloatBuffer(Source, Destination, Count);
#else
    // Check if the Tensor is long enough to use threads.
    // Check if the Thread Pool is available.
    // If not, execute single threaded conversion of half to float
    if (!((Count > MLAS_MIN_TENSOR_SIZE_FOR_HALF_TO_FLOAT_CONVERSION_IN_PARALLEL) && ThreadPool)) {
        MlasConvertHalfToFloatBuffer(Source, Destination, Count);
    }
    else {

        // Calculate the number of compute cycles per implementation
        size_t num_compute_cycles;
        if (MLAS_CPUIDINFO::GetCPUIDInfo().HasSSE3()) {
            num_compute_cycles = Count >> 1;
        } else if (MLAS_CPUIDINFO::GetCPUIDInfo().HasAVX2()) {
            num_compute_cycles = Count >> 2;
        } else {
            num_compute_cycles = Count * 10;
        }

        MLAS_THREADPOOL::TryParallelFor(
            ThreadPool, Count,
            // Tensor Op Cost
            {
                static_cast<double>(Count * sizeof(MLAS_FP16)),  // Size of no. of elements in bytes to be loaded
                static_cast<double>(Count * sizeof(float)),      // Size of no. of elements in bytes to be stored
                static_cast<double>(num_compute_cycles),         // No. of compute cycles required for the tensor op
            },
            // Lambda function required by TryParallelFor method
            [Source, Destination](std::ptrdiff_t first_span, std::ptrdiff_t last_span) {
                MlasConvertHalfToFloatBuffer(
                    Source + first_span,
                    Destination + first_span,
                    static_cast<size_t>(last_span - first_span));
            }
        );
    }
#endif // BUILD_MLAS_NO_ONNXRUNTIME
}

void
MLASCALL
MlasConvertFloatToHalfBuffer(
    const float* Source,
    MLAS_FP16* Destination,
    size_t Count
)
{
    if (GetMlasPlatform().CastF32ToF16Kernel == nullptr) {
        for (size_t i = 0; i < Count; ++i) {
            Destination[i] = MLAS_FP16(Source[i]);
        }
    } else {
        // If the kernel is available, use it to perform the conversion.
        GetMlasPlatform().CastF32ToF16Kernel(Source, reinterpret_cast<unsigned short*>(Destination), Count);
    }
}
