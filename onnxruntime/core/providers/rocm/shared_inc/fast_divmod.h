// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <limits>
#include <hip/hip_runtime.h>
#include <cmath>
#include "core/common/common.h"

namespace onnxruntime {
namespace rocm {

// The code below is based on section 4 Unsigned division of paper https://gmplib.org/~tege/divcnst-pldi94.pdf
// In current ORT, fast_divmod is used for calculating the position of a element in tensor,
// so unsigned integer division from the paper is good enough for ORT. The advantage is that div is very simple,
// then GPU compiler can do loop unroll easilly when divmod is called in a loop.
struct fast_divmod {
  fast_divmod(int d = 1) {
    d_ = d == 0 ? 1 : d;
    ORT_ENFORCE(d_ >= 1 && d_ <= static_cast<uint32_t>(std::numeric_limits<int>::max()));

    for (l_ = 0; l_ < 32; l_++) if ((1U << l_) >= d_) break;

    uint64_t one = 1;
    uint64_t m = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
    M_ = static_cast<uint32_t>(m);
    // according to paper, the value of m' should fit in a unsigned integer.
    ORT_ENFORCE(M_ > 0 && M_ == m);
  }

  __host__ __device__ inline int div(int n) const {
#ifdef __HIP_DEVICE_COMPILE__
    uint32_t t = __umulhi(M_, n);
    return (t + n) >> l_;
#else
    // Using uint64_t for t, then t + n won't overflow.
    uint64_t t = ((uint64_t) M_ * n) >> 32;
    return static_cast<int>((t + n) >> l_);
#endif
  }

  __host__ __device__ inline int mod(int n) const {
    return n - div(n) * d_;
  }

  __host__ __device__ inline void divmod(int n, int& q, int& r) const {
    q = div(n);
    r = n - q * d_;
  }

  uint32_t d_;  // divisor
  uint32_t M_;  // m' in the paper.
  uint32_t l_;  // l_ = ceil(log2(d_))
};

}  // namespace rocm
}  // namespace onnxruntime
