// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/util.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

int CeilingDivision(int n, int m) {
  int r = (n - 1) / m + 1;
  return r;
}

Timer::Timer() {
  HIP_CHECK(hipEventCreate(&start_));
  HIP_CHECK(hipEventCreate(&end_));
}

void Timer::Start() {
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(start_, nullptr));
}

void Timer::End() {
  HIP_CHECK(hipEventRecord(end_, nullptr));
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

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
