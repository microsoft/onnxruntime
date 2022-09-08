// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/tunable/tunable.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {

Timer::Timer(hipStream_t stream) : stream_(stream) {
  HIP_CHECK(hipEventCreate(&start_));
  HIP_CHECK(hipEventCreate(&end_));
}

void Timer::Start() {
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(start_, stream_));
}

void Timer::End() {
  HIP_CHECK(hipEventRecord(end_, stream_));
  HIP_CHECK(hipEventSynchronize(end_));
}

float Timer::Duration() {
  float time;
  // time is in ms with a resolution of 1 us
  HIP_CHECK(hipEventElapsedTime(&time, start_, end_));
  return time;
}

Timer::~Timer() {
  HIP_CHECK(hipEventDestroy(start_));
  HIP_CHECK(hipEventDestroy(end_));
}

}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
