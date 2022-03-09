#pragma once

// FIXME: this kernel only copies if the width % 4 == 0, otherwise, weird thing
// might happen
__kernel void CopyBuffer2DToImage2D(
    __global const float4* input,
    __write_only image2d_t output,
    __private const int width,
    __private const int height) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x < width && y < height) {
    write_imagef(output, (int2)(x, y), input[x + y * width]);
  }
}
