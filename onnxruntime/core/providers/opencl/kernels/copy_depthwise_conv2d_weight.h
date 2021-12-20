#pragma once

#include "./utils.h"

// C_i == input_channel_per_group, since it is depthwise, it is always 1
// C_o == total_output_channel == group * output_channel_per_group, currently,
//     we are limiting output_channel_per_group == 1
__kernel void CopyDepthwiseConvWeightBufferToImage(
    const int gs_dim0, // gs_dim0 = output width  = C_i*K_h*K_w, where C_i == 1
    const int gs_dim1, // gs_dim1 = output height = CeilDiv(C_o, 4)
    __global const float* data,
    __private const int4 kernel_shape,
    __private const int CiHW,  // C_i * K_h * K_w, where C_i == 1
    __write_only image2d_t output) {
#define C_o kernel_shape.s0
#define C_i kernel_shape.s1
#define K_h kernel_shape.s2
#define K_w kernel_shape.s3
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= gs_dim0 || y >= gs_dim1) return;

  const int co = y * 4;
  const int kh = x / K_w;
  const int kw = x % K_w;

  // (C_i*K_h*K_w)*co (K_h*K_w)*ci + K_w*kh + kw, ci == 0, C_i == 1, then
  // (K_h*K_w)*co + K_w*kh + kw ==> (K_h*co + kh)*K_w + kw
  const int base_offset = mad24(mad24(co, K_h, kh), K_w, kw);

  float4 v = 0;
  SAFE_GATHER_LDG_VEC4(v, data, base_offset, CiHW, C_o - co);
  WI_F(output, (int2)(x, y), CONVERT_FLOAT4(v));
#undef K_w
#undef K_h
#undef C_i
#undef C_o
}
