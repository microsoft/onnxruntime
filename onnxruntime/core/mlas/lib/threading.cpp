/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    threading.cpp

Abstract:

    This module implements platform specific threading support.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_WIN32_THREADPOOL)

//
// Define the parameters to execute threaded work using the Windows thread pool
// library.
//

struct MLAS_THREADED_WORK_BLOCK {
    volatile LONG Counter;
    PMLAS_THREADED_ROUTINE ThreadedRoutine;
    void* Context;
};

void
CALLBACK
MlasThreadedWorkCallback(
    PTP_CALLBACK_INSTANCE Instance,
    void* Context,
    PTP_WORK WorkObject
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute one iteration of a
    batch of threaded work.

Arguments:

    Instance - Supplies the callback instance object.

    Context - Supplies the pointer to the parameters for the operation.

    WorkObject - Supplies the threadpool work object.

Return Value:

    None.

--*/
{
    MLAS_UNREFERENCED_PARAMETER(Instance);
    MLAS_UNREFERENCED_PARAMETER(WorkObject);

    MLAS_THREADED_WORK_BLOCK* WorkBlock = (MLAS_THREADED_WORK_BLOCK*)Context;

    LONG Index = InterlockedIncrement(&WorkBlock->Counter) - 1;

    WorkBlock->ThreadedRoutine(WorkBlock->Context, Index);
}

#endif

void
MlasExecuteThreaded(
    MLAS_THREADED_ROUTINE ThreadedRoutine,
    void* Context,
    int32_t Iterations,
    ThreadPool *ExternalThreadPool
    )
{
    //
    // Execute the routine directly if only one iteration is specified.
    //

    if (Iterations == 1) {
        ThreadedRoutine(Context, 0);
        return;
    }

    //
    // Use an external thread pool if one is provided.
    //

    if (!(ExternalThreadPool == nullptr)) {
      std::function<void(int)> WorkObject = [&](int32_t tid) { ThreadedRoutine(Context, tid); };
      ExternalThreadPool->ParallelFor(Iterations, WorkObject);

      return;
    }

#if defined(MLAS_USE_WIN32_THREADPOOL)

    //
    // Schedule the threaded iterations using a work object.
    //

    MLAS_THREADED_WORK_BLOCK WorkBlock;

    PTP_WORK WorkObject = CreateThreadpoolWork(MlasThreadedWorkCallback, &WorkBlock, nullptr);

    if (WorkObject != nullptr) {

        WorkBlock.Counter = 0;
        WorkBlock.ThreadedRoutine = ThreadedRoutine;
        WorkBlock.Context = Context;

        for (int32_t tid = 1; tid < Iterations; tid++) {
            SubmitThreadpoolWork(WorkObject);
        }

        //
        // Execute the remaining iteration on this thread.
        //

        ThreadedRoutine(Context, Iterations - 1);

        //
        // Wait for the work object callbacks to complete.
        //

        WaitForThreadpoolWorkCallbacks(WorkObject, FALSE);
        CloseThreadpoolWork(WorkObject);

        return;
    }

    //
    // Fallback to a serialized implementation.
    //

#endif

    //
    // Execute the routine for the specified number of iterations.
    //
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int32_t tid = 0; tid < Iterations; tid++) {
        ThreadedRoutine(Context, tid);
    }
}
