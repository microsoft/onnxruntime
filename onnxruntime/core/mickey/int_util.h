/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    int_util.h
 *
 * Abstract:
 *   Utils for integer operations, mostly constants and bit manipulation.
 */

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

namespace mickey {

CUTLASS_HOST_DEVICE
constexpr int div_up(int a, int b) {
  return (a + b - 1) / b;
}

CUTLASS_HOST_DEVICE
constexpr int round_up(int a, int b) {
  return div_up(a, b) * b;
}

CUTLASS_HOST_DEVICE
constexpr int log2(int a) {
  return (a > 1) ? 1 + log2(a >> 1) : 0;
}

template <int N>
CUTLASS_HOST_DEVICE constexpr int div_power2(int a) {
  static_assert((N & (N - 1)) == 0, "To use div_power2<N>, N must be a power of 2.");
  constexpr int log2_N = log2(N);
  return a >> log2_N;
}

template <int N>
CUTLASS_HOST_DEVICE constexpr int mod_power2(int a) {
  static_assert((N & (N - 1)) == 0, "To use mod_power2<N>, N must be a power of 2.");
  return a & (N - 1);
}

template <int N>
CUTLASS_HOST_DEVICE constexpr int mul_power2(int a) {
  static_assert((N & (N - 1)) == 0, "To use mul_power2<N>, N must be a power of 2.");
  constexpr int log2_N = log2(N);
  return a << log2_N;
}

}  // namespace mickey
