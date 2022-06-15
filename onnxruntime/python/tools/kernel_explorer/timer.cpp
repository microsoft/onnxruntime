// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "timer.h"

Timer::Timer() {
  hipEventCreate(&start_);
  hipEventCreate(&end_);
}

void Timer::Start() {
  hipDeviceSynchronize();
  hipEventRecord(start_, nullptr);
}

void Timer::End() {
  hipEventRecord(end_, nullptr);
  hipEventSynchronize(end_);
}

float Timer::time() {
  float time;
  // time is in ms with a resolution of 1 us
  hipEventElapsedTime(&time, start_, end_);
  return time;
}

Timer::~Timer() {
  hipEventDestroy(start_);
  hipEventDestroy(end_);
}
