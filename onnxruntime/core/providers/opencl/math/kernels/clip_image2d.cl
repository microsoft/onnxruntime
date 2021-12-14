// TODO: use transformer to fuse clip into Conv via relu6 activation
#include <kernels/utils.h>

__kernel void ClipNCHW(
    const int gs_dim0,
    const int gs_dim1,
    __read_only image2d_t data,  // width = CeilDiv(C, 4) * W, height = N*H
    __write_only image2d_t output,
    __private const int C, // tensor's C size
    __private const int W, // tensor's W size
    __private const float min,
    __private const float max) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= gs_dim0 || y >= gs_dim1) return;

  FLOAT4 v = RI_F(data, (int2)(x, y));
  const int c = (x / W) * 4;

  int remain = C - c;
  FLOAT4 clipped = clamp(v, min, max);
  if (remain >= 4) {
    WI_F(output, (int2)(x, y), clipped);
  } else if (remain == 3) {
    FLOAT4 selected = select(v, clipped, MASK3);
    WI_F(output, (int2)(x, y), selected);
  } else if (remain == 2) {
    FLOAT4 selected = select(v, clipped, MASK2);
    WI_F(output, (int2)(x, y), selected);
  } else if (remain == 1) {
    FLOAT4 selected = select(v, clipped, MASK1);
    WI_F(output, (int2)(x, y), selected);
  }
}
