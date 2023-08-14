// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/tunable/util.h"

#include "core/providers/rocm/shared_inc/rocm_call.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {

Timer::Timer(hipStream_t stream) : TimerBase(stream) {
  HIP_CALL_THROW(hipEventCreate(&start_));
  HIP_CALL_THROW(hipEventCreate(&end_));
}

void Timer::Start() {
  HIP_CALL_THROW(hipDeviceSynchronize());
  HIP_CALL_THROW(hipEventRecord(start_, stream_));
}

void Timer::End() {
  HIP_CALL_THROW(hipEventRecord(end_, stream_));
  HIP_CALL_THROW(hipEventSynchronize(end_));
}

float Timer::Duration() {
  float time;
  // time is in ms with a resolution of 1 us
  HIP_CALL_THROW(hipEventElapsedTime(&time, start_, end_));
  return time;
}

Timer::~Timer() {
  HIP_CALL_THROW(hipEventDestroy(start_));
  HIP_CALL_THROW(hipEventDestroy(end_));
}

}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
