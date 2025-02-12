# Notes on Threading in ORT

# Deprecation warning : openmp thread support is removed from ORT

This document is intended for ORT developers.

ORT allows the usage of either OpenMP or non-OpenMP (ORT) threads for execution. Threadpool management
is abstracted behind: (1) ThreadPool class in [threadpool.h](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/platform/threadpool.h) and (2) functions in [thread_utils.h](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/util/thread_utils.h).

When developing an op, please use these abstractions to parallelize your code. These abstractions centralize 2 things.
When OpenMP is enabled, they resort to using OpenMP. When OpenMP is disabled they resort to sequential execution if the threadpool ptr is NULL or schedule the tasks on the threadpool otherwise.

Examples of these abstractions are: ([threadpool.h](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/platform/threadpool.h) has more documentation for these)
* TryParallelFor
* TrySimpleParallelFor
* TryBatchParallelFor
* ShouldParallelize
* DegreeOfParallelism

These static methods abstract over the different implementation choices.  They can run over the ORT thread pool, or run over OpenMP, or run sequentially.

In addition, ThreadPool::ParallelSection allows a series of loops to
be grouped together in a single parallel section. This allows an
operator to amortize loop entry/exit costs in cases where it is
impractical to refactor code into a single large loop.

**Please do not write #ifdef pragma omp in operator code**.

For intra op parallelism ORT users can use either OpenMP or ORT threadpool. The choice of using OpenMP is indicated by building ORT with ```--use_openmp``` switch. For inter op parallelism, however, we always use the ORT threadpool.
