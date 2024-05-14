// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tunable/util.h"

#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {
namespace cuda {
namespace tunable {

Timer::Timer(cudaStream_t stream) : TimerBase(stream) {
  CUDA_CALL_THROW(cudaEventCreate(&start_));
  CUDA_CALL_THROW(cudaEventCreate(&end_));
}

void Timer::Start() {
  CUDA_CALL_THROW(cudaDeviceSynchronize());
  CUDA_CALL_THROW(cudaEventRecord(start_, stream_));
}

void Timer::End() {
  CUDA_CALL_THROW(cudaEventRecord(end_, stream_));
  CUDA_CALL_THROW(cudaEventSynchronize(end_));
}

float Timer::Duration() {
  float time;
  // time is in ms with a resolution of 1 us
  CUDA_CALL_THROW(cudaEventElapsedTime(&time, start_, end_));
  return time;
}

Timer::~Timer() {
  CUDA_CALL_THROW(cudaEventDestroy(start_));
  CUDA_CALL_THROW(cudaEventDestroy(end_));
}

}  // namespace tunable
}  // namespace cuda
}  // namespace onnxruntime
