# Notes on Threading in ORT

This document is intended for ORT developers.

ORT allows the usage of either OpenMP or non-OpenMP (ORT) threads for execution. Threadpool management
is abstracted behind: (1) ThreadPool class in [threadpool.h](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/platform/threadpool.h) and (2) functions in [thread_utils.h](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/util/thread_utils.h).

When developing an op, please use these abstractions to parallelize your code. These abstractions centralize 2 things.
When OpenMP is enabled, they resort to using OpenMP. When OpenMP is disabled they resort to sequential execution if the threadpool ptr is NULL or schedule the tasks on the threadpool otherwise.

Examples of these abstractions are: ([threadpool.h](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/platform/threadpool.h) has more documentation for these)
* TryBatchParallelFor
* TryParallelFor
* TrySimpleParallelFor
* DegreeOfParallelism

**Please do not write #ifdef pragma omp in operator code**.

For intra op parallelism ORT users can use either OpenMP or ORT threadpool. The choice of using OpenMP is indicated by building ORT with ```--use_openmp``` switch. For inter op parallelism, however, we always use the ORT threadpool.
