#pragma once

__kernel void CopyBuffer1DToImage2D(
    int width, int height,
    __global const float* data,
    int nelem,  // num of float elements
    __write_only image2d_t output) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int idx = mad24(width, y, x) * 4;
  int remain = nelem - idx;
  float4 v = 0;  // NOTE: buffer r/w always assume fp32
  if (idx < nelem) {
    if (remain >= 4) {
      __global const float4* data4 = (__global const float4*)data;
      v = data4[idx / 4];
    } else if (remain == 3) {
      v.x = data[idx + 0];
      v.y = data[idx + 1];
      v.z = data[idx + 2];
    } else if (remain == 2) {
      v.x = data[idx + 0];
      v.y = data[idx + 1];
    } else if (remain == 1) {
      v.x = data[idx];
    }
    WI_F(output, (int2)(x, y), CONVERT_FLOAT4(v));
  }
}

__kernel void CopyImage2DToBuffer1D(
    int width, int height,
    __read_only image2d_t data,
    __global float* output,
    int nelem  // number of float elements
) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int idx = mad24(width, y, x) * 4;
  int remain = nelem - idx;
  if (idx < nelem) {
    float4 v = convert_float4(RI_F(data, (int2)(x, y)));  // NOTE: buffer r/w always assume fp32
    if (remain >= 4) {
      __global float4* output4 = (__global float4*)output;
      output4[idx / 4] = v;
    } else if (remain == 3) {
      output[idx + 0] = v.x;
      output[idx + 1] = v.y;
      output[idx + 2] = v.z;
    } else if (remain == 2) {
      output[idx + 0] = v.x;
      output[idx + 1] = v.y;
    } else if (remain == 1) {
      output[idx] = v.x;
    }
  }
}
