/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    threading.cpp

Abstract:

    This module implements platform specific threading support.

--*/

#include "mlasi.h"

void
MlasExecuteThreaded(
    MLAS_THREADED_ROUTINE* ThreadedRoutine,
    void* Context,
    ptrdiff_t Iterations,
    MLAS_THREADPOOL* ThreadPool
    )
{
    //
    // Execute the routine directly if only one iteration is specified.
    //

    if (Iterations == 1) {
        ThreadedRoutine(Context, 0);
        return;
    }

#if defined(MLAS_NO_ONNXRUNTIME_THREADPOOL)
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    //
    // Fallback to OpenMP or a serialized implementation.
    //

    //
    // Execute the routine for the specified number of iterations.
    //
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (ptrdiff_t tid = 0; tid < Iterations; tid++) {
        ThreadedRoutine(Context, tid);
    }
#else
    //
    // Schedule the threaded iterations using the thread pool object.
    //

    MLAS_THREADPOOL::TrySimpleParallelFor(ThreadPool, Iterations, [&](ptrdiff_t tid) {
        ThreadedRoutine(Context, tid);
    });
#endif
}
