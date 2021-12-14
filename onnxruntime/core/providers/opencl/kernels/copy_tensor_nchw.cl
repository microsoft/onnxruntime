// FIXME: LICENSE NOTICE:
// adapted from TNN original BSD3.
#include <kernels/utils.h>

// launch a grid with total [X = RoundToMultiple(width, k), Y = height] threads,
// height is not rounded up because threads only continuous in x dimension.
__kernel void CopyBufferNCHWToImage2D(
    const int width, const int height,  // image, width = CeilDiv(C,4)*W, height = N*H
    __global const float* data,
    /*const int N,*/ const int C, const int H, const int W,
    __write_only image2d_t output) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < width && y < height) {
    const int n = y / H;
    const int h = y % H;
    const int w = x % W;
    const int c = (x / W) * 4;  // every thread handle 4 elements, gather read, vector write

    // indexing into the NCHW data
    const int HW = H * W;
    const int base_index = (C * n + c) * HW + W * h + w;

    // channel is not consecutive in memory
    const int remain_channel = C - c;
    float4 v = 0;  // NOTE: buffer r/w always assume fp32
    SAFE_GATHER_LDG_VEC4(v, data, base_index, HW, remain_channel);
    WI_F(output, (int2)(x, y), CONVERT_FLOAT4(v));
  }
}

// launch a grid with total [X = RoundToMultiple(width, k), Y = height] threads
__kernel void CopyImage2DToBufferNCHW(
    int width, int height,  // width = CeilDiv(C,4)*W, height = N*H
    __read_only image2d_t data,
    __global float* output,
    /*int N,*/ int C, int H, int W) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < width && y < height) {
    const int n = y / H;
    const int h = y % H;
    const int w = x % W;
    const int c = (x / W) * 4;  // every thread handle 4 elements, vector read, scatter write

    // indexing into the NCHW data
    const int HW = H * W;
    const int base_index = (C * n + c) * HW + W * h + w;
    const float4 v = convert_float4(RI_F(data, (int2)(x, y)));

    const int remain_channel = C - c;
    // NOTE: buffer r/w always assume fp32
    SAFE_SCATTER_STG_VEC4(output, base_index, HW, remain_channel, v);
  }
}
