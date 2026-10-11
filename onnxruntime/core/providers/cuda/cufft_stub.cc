// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Provides definitions for the small set of cuFFT entry points used by the CUDA
// Execution Provider. Each stub forwards to the matching symbol resolved from the
// dynamically loaded cuFFT runtime library. Linking against these stubs (instead
// of the cuFFT import library) removes the hard runtime dependency on cuFFT so
// that models without FFT (Rfft/Irfft) ops can run without cuFFT installed.

#include "core/providers/cuda/cufft_loader.h"

#ifndef USE_CUDA_MINIMAL

#include "cufft.h"
#include "cufftXt.h"

#define ORT_CUFFT_FORWARD_RESULT(name, ...)                            \
  using Fn = decltype(&name);                                          \
  auto fn = onnxruntime::cuda::CufftLibrary::Get().Resolve<Fn>(#name); \
  return fn != nullptr ? fn(__VA_ARGS__) : CUFFT_INTERNAL_ERROR

extern "C" {

cufftResult CUFFTAPI cufftCreate(cufftHandle* handle) {
  ORT_CUFFT_FORWARD_RESULT(cufftCreate, handle);
}

cufftResult CUFFTAPI cufftDestroy(cufftHandle plan) {
  ORT_CUFFT_FORWARD_RESULT(cufftDestroy, plan);
}

cufftResult CUFFTAPI cufftSetStream(cufftHandle plan, cudaStream_t stream) {
  ORT_CUFFT_FORWARD_RESULT(cufftSetStream, plan, stream);
}

cufftResult CUFFTAPI cufftXtMakePlanMany(cufftHandle plan, int rank, long long int* n,
                                         long long int* inembed, long long int istride, long long int idist,
                                         cudaDataType inputtype,
                                         long long int* onembed, long long int ostride, long long int odist,
                                         cudaDataType outputtype,
                                         long long int batch, size_t* workSize, cudaDataType executiontype) {
  ORT_CUFFT_FORWARD_RESULT(cufftXtMakePlanMany, plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride,
                           odist, outputtype, batch, workSize, executiontype);
}

cufftResult CUFFTAPI cufftXtExec(cufftHandle plan, void* input, void* output, int direction) {
  ORT_CUFFT_FORWARD_RESULT(cufftXtExec, plan, input, output, direction);
}

}  // extern "C"

#undef ORT_CUFFT_FORWARD_RESULT

#endif  // USE_CUDA_MINIMAL
