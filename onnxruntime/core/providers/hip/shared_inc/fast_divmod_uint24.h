//
// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <hip/hip_runtime.h>
#include <cmath>

namespace onnxruntime {
namespace hip {

///////////////////uint24///////////////////////////////
template<typename T>
struct __attribute__((packed)) uint24_t {
    T x : 24;
};

template<typename T>
inline __host__ __device__
T mul24(T x, T y) noexcept
{
    return uint24_t<T>{x}.x * uint24_t<T>{y}.x;
}

template long long mul24<long long>(long long, long long);

template<typename T>
inline __host__ __device__
T mad24(T x, T y, T z) noexcept
{
    return uint24_t<T>{x}.x * uint24_t<T>{y}.x + uint24_t<T>{z}.x;
}

template long long mad24<long long>(long long, long long, long long);

template<typename T>
inline __host__ __device__
T div24(T x, T y) noexcept
{
    return uint24_t<T>{x}.x / uint24_t<T>{y}.x;
}

template long long div24<long long>(long long, long long);

template<typename T>
inline __host__ __device__
T mod24(T x, T y) noexcept
{
    return uint24_t<T>{x}.x % uint24_t<T>{y}.x;
}

template long long mod24<long long>(long long, long long);

struct fast_divmod {
  fast_divmod(int d = 1) : d_(d) {
  }

  __host__ __device__ inline int div(int n) const {
    return div24(n, d_);
  }

  __host__ __device__ inline int mod(int n) const {
    return mod24(n, d_);
  }

  __host__ __device__ inline void divmod(int n, int& q, int& r) const {
    q = int(div24(n, d_) & 0x7FFFFF);
    r = n - int(mul24(q, d_) & 0x7FFFFF);
  }

  int d_;  // d above.
};

}  // namespace hip
}  // namespace onnxruntime
