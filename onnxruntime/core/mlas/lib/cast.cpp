/*++

Copyright (c) Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    cast.cpp

Abstract:

    This module implements Half (F16) to Single (F32) precision casting.

--*/
#include "mlasi.h"
#include "core/platform/threadpool.h"

union fp32_bits {
    uint32_t u;
    float f;
};

static size_t
GetChunkSizePerThread(size_t Count, size_t num_threads, size_t multiple)
{
    // distribute data to multiple (for single instruction)
    size_t blks = MlasDivRoundup(Count, multiple);
    // distribute blks to threads
    size_t blks_per_thread = MlasDivRoundup(blks, num_threads);
    size_t chunk_size = blks_per_thread * multiple;
    return chunk_size;
}

void
MLASCALL
MlasConvertHalfToFloatBuffer(
    const MLAS_FP16* Source,
    float* Destination,
    size_t Count,
    MLAS_THREADPOOL* thread_pool
    )
{
    if (GetMlasPlatform().CastF16ToF32Kernel == nullptr) {
        for (size_t i = 0; i < Count; ++i) {
            Destination[i] = Source[i].ToFloat();
        }
    } else {
        // If the kernel is available, use it to perform the conversion.
        if (thread_pool == nullptr) {
            GetMlasPlatform().CastF16ToF32Kernel(reinterpret_cast<const unsigned short*>(Source), Destination, Count);
        } else {
            size_t num_threads = 8;
            size_t multiple = 16;
            size_t chunk_size = GetChunkSizePerThread(Count, num_threads, multiple);
            onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, num_threads, [&](ptrdiff_t tid) {
                int start = static_cast<int>(tid * chunk_size);
                int size = static_cast<int>((start + chunk_size > Count) ? Count - start : chunk_size);
                if (size > 0) {
                    const unsigned short* per_thread_source = reinterpret_cast<const unsigned short*>(Source) + start;
                    float* per_thread_destination = Destination + start;
                    GetMlasPlatform().CastF16ToF32Kernel(per_thread_source, per_thread_destination, size);
                }
            });
        }
    }
}

void
MLASCALL
MlasConvertFloatToHalfBuffer(
    const float* Source,
    MLAS_FP16* Destination,
    size_t Count,
    MLAS_THREADPOOL* thread_pool
)
{
    if (GetMlasPlatform().CastF32ToF16Kernel == nullptr) {
        for (size_t i = 0; i < Count; ++i) {
            Destination[i] = MLAS_FP16(Source[i]);
        }
    } else {
        // If the kernel is available, use it to perform the conversion.
        if (thread_pool == nullptr) {
            GetMlasPlatform().CastF32ToF16Kernel(Source, reinterpret_cast<unsigned short*>(Destination), Count);
        } else {
            size_t num_threads = 8;
            size_t multiple = 16;
            size_t chunk_size = GetChunkSizePerThread(Count, num_threads, multiple);
            onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, num_threads, [&](ptrdiff_t tid) {
                int start = static_cast<int>(tid * chunk_size);
                int size = static_cast<int>((start + chunk_size > Count) ? Count - start : chunk_size);
                if (size > 0) {
                    const float* per_thread_source = Source + start;
                    unsigned short* per_thread_destination = reinterpret_cast<unsigned short*>(Destination) + start;
                    GetMlasPlatform().CastF32ToF16Kernel(per_thread_source, per_thread_destination, size);
                }
            });
        }
    }
}
