/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Portions Copyright (c) Microsoft Corporation

#include <sys/time.h>
#include <ctime>
#include <cstring>
#include "core/platform/env_time.h"

namespace onnxruntime {

namespace {

class PosixEnvTime : public EnvTime {
 public:
  PosixEnvTime() = default;

  uint64_t NowMicros() override {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
  }
};

}  // namespace

#if defined(PLATFORM_POSIX) || defined(__ANDROID__) || defined(ENABLE_ORT_WASM)
EnvTime* EnvTime::Default() {
  static PosixEnvTime default_env_time;
  return &default_env_time;
}
#endif

bool GetMonotonicTimeCounter(TIME_SPEC* value) {
  return clock_gettime(CLOCK_MONOTONIC, value) == 0;
}

void SetTimeSpecToZero(TIME_SPEC* value) {
  memset(value, 0, sizeof(TIME_SPEC));
}

void AccumulateTimeSpec(TIME_SPEC* base, const TIME_SPEC* p_y, const TIME_SPEC* p_x) {
  TIME_SPEC y_l = *p_y;
  TIME_SPEC x_l = *p_x;
  TIME_SPEC* y = &y_l;
  TIME_SPEC* x = &x_l;
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_nsec < y->tv_nsec) {
    long nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_nsec - y->tv_nsec > 1000000000) {
    long nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
    y->tv_nsec += 1000000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_nsec is certainly positive. */
  base->tv_sec += x->tv_sec - y->tv_sec;
  base->tv_nsec += x->tv_nsec - y->tv_nsec;
  if (base->tv_nsec >= 1000000000) {
    base->tv_nsec -= 1000000000;
    ++base->tv_sec;
  }
}

//Return the interval in seconds.
//If the function fails, the return value is zero
double TimeSpecToSeconds(const TIME_SPEC* value) {
  return value->tv_sec + value->tv_nsec / static_cast<double>(1000000000);
}

}  // namespace onnxruntime
