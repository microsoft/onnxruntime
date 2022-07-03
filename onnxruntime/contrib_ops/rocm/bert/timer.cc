// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/timer.h"

#define HIP_CHECKED_CALL(expr)                                                             \
  do {                                                                                     \
    auto status = expr;                                                                    \
    if (status != hipSuccess) {                                                            \
      std::printf("HIP Error at %s:%d\n    Error name  : %s\n    Error string: %s\n",      \
                  __FILE__, __LINE__, hipGetErrorName(status), hipGetErrorString(status)); \
      std::abort();                                                                        \
    }                                                                                      \
  } while (0)

namespace onnxruntime {
namespace contrib {
namespace rocm {

Timer::Timer() {
  HIP_CHECKED_CALL(hipEventCreate(&start_));
  HIP_CHECKED_CALL(hipEventCreate(&end_));
}

void Timer::Start() {
  HIP_CHECKED_CALL(hipDeviceSynchronize());
  HIP_CHECKED_CALL(hipEventRecord(start_, nullptr));
}

void Timer::End() {
  HIP_CHECKED_CALL(hipEventRecord(end_, nullptr));
  HIP_CHECKED_CALL(hipEventSynchronize(end_));
}

float Timer::time() {
  float time;
  // time is in ms with a resolution of 1 us
  HIP_CHECKED_CALL(hipEventElapsedTime(&time, start_, end_));
  return time;
}

Timer::~Timer() {
  HIP_CHECKED_CALL(hipEventDestroy(start_));
  HIP_CHECKED_CALL(hipEventDestroy(end_));
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
