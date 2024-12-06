/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qsoftmax_kernel_avx2.cpp

Abstract:

    This module implements the lookup-based quantized softmax kernels,
    it's a naive implementation and works for all arch.


--*/

#include <algorithm>
#include <cassert>
#include <utility>

#include "mlasi.h"

void MLASCALL MlasQuantizeSoftmaxU8KernelNaive(size_t D, const uint8_t* x_data, uint8_t* y_data,
                                               const float* lookup_table, float y_scale, uint8_t yzp, float*) {
  constexpr size_t N = 1;
  const auto c_y_scale = y_scale;
  const auto c_y_zp = yzp;
  const uint8_t* x_t = x_data + 0 * D;
  uint8_t* y_t = y_data + 0 * D;
  for (size_t first = 0; first < N; first++) {
    // reduceMaxUint8
    uint8_t xmax = *std::max_element(x_t, x_t + D);
    // we want the xmas to align with 255 for higher precision.
    // as we build a lookup table with X-255. So we could use the adjustment here
    // to let all numbers have a shift in the lookup table.
    // 1 2 3 4 5 ...........................254 255
    // 1   3   5 ... 10
    // after the shift --->
    //                        235  237  239  .. 255
    const float* shifted_lookuptable = lookup_table + 255 - xmax;
    size_t elements_n = D;
    // reduceSumUin8ToUint32: need speedup
    // vsum = \sum_i{e^x_i}
    float vsum = 0;
    const uint8_t* x_t_cur = x_t;
    do {
      const size_t vx = *x_t_cur++;
      vsum += shifted_lookuptable[vx];
    } while (--elements_n != 0);
    if (vsum == 0) {
      return;
    }
    elements_n = D;
    x_t_cur = x_t;
    // elementwise div, y_i=\frac{x_i}{vsum}
    do {
      const size_t vx = *x_t_cur++;
      const float vt = shifted_lookuptable[vx];
      // simulate round function, and re-quant to uint8
      const uint32_t vq = static_cast<uint32_t>(std::nearbyintf((vt * c_y_scale) / vsum)) + c_y_zp;
      const uint8_t vy = vq > 255 ? static_cast<uint8_t>(255) : static_cast<uint8_t>(vq);
      *y_t++ = vy;
    } while (--elements_n != 0);
    x_t = x_t_cur;
  }
}

void MLASCALL MlasQuantizeSoftmaxI8KernelNaive(size_t D, const int8_t* x_data, int8_t* y_data,
                                               const float* lookup_table, float y_scale, int8_t yzp, float*) {
  constexpr size_t N = 1;
  const auto c_y_scale = y_scale;
  const auto c_y_zp = yzp;
  size_t first = 0;
  const int8_t* x_t = x_data + first * D;
  int8_t* y_t = y_data + first * D;
  for (; first < N; first++) {
    // reduceMaxInt8
    int8_t xmax = *std::max_element(x_t, x_t + D);
    const int32_t adjustment = int32_t(127) - xmax;
    const float* shifted_lookuptable = lookup_table;
    size_t elements_n = D;
    // reduceSumUin8ToUint32: need speedup
    float vsum = 0;
    const int8_t* x_t_cur = x_t;
    do {
      const uint8_t vx = uint8_t(adjustment + (*x_t_cur++));
      vsum += shifted_lookuptable[vx];
    } while (--elements_n != 0);
    if (vsum == 0) {
      return;
    }
    elements_n = D;
    x_t_cur = x_t;
    // elementwise div
    do {
      const uint8_t vx = uint8_t(adjustment + (*x_t_cur++));
      const float vt = shifted_lookuptable[vx];
      // simulate round function, and re-quant to Int8
      const int32_t vq = static_cast<int32_t>(std::nearbyintf(((vt * c_y_scale)) / vsum)) + c_y_zp;
      const int8_t vy = static_cast<int32_t>(vq) > 255 ? static_cast<int8_t>(255) : static_cast<int8_t>(vq);
      *y_t++ = vy;
    } while (--elements_n != 0);
    x_t = x_t_cur;
  }
}
