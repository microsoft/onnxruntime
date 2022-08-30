// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <iostream>

#define HIP_CHECK(expr)                      \
  do {                                       \
    auto status = expr;                      \
    if (status != hipSuccess) {              \
      std::cerr << hipGetErrorName(status);  \
      std::abort();                          \
    }                                        \
  } while (0)

namespace onnxruntime {
namespace contrib {
namespace rocm {

int CeilingDivision(int n, int m);

template<typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) AlignedVector {
  T val[VecSize];
};

class Timer {
 public:
  Timer();
  void Start();
  void End();
  float Duration();
  ~Timer();

 private:
  hipEvent_t start_;
  hipEvent_t end_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
