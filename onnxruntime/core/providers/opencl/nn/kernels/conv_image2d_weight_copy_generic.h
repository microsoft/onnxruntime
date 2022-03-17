// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels/utils.h"

__kernel void CopyGenericConv2DWeightBufferToImage(
    const int width, const int height,  // image, width = C_i, height = CeilDiv(C_o, 4)*K_h*K_w
    __global const float* data,
    __private const int4 kernel_shape,
    __private const int HW,  // K_h * K_w
    __write_only image2d_t output) {
#define C_o kernel_shape.s0
#define C_i kernel_shape.s1
#define K_h kernel_shape.s2
#define K_w kernel_shape.s3
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;

  const int ci = x;
  const int co = (y / HW) * 4;
  const int hw = y % HW;
  const int kh = hw / K_w;
  const int kw = hw % K_w;

  const int CiHW = mul24(C_i, HW);
  // (C_i*K_h*K_w)*co + (K_h*K_w)*ci + K_w*kh + kw
  const int base_offset = mad24(CiHW, co, mad24(HW, ci, mad24(K_w, kh, kw)));

  float4 v = 0;
  SAFE_GATHER_LDG_VEC4(v, data, base_offset, CiHW, C_o - co);
  WI_F(output, (int2)(x, y), CONVERT_FLOAT4(v));
#undef K_w
#undef K_h
#undef C_i
#undef C_o
}
