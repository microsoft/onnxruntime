// TODO: use transformer to fuse clip into Conv via relu6 activation

__kernel void Clip(
    const int gs_dim0,
    const int gs_dim1,
    __read_only image2d_t data,
    __write_only image2d_t output,
    __private const float min,
    __private const float max) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= gs_dim0 || y >= gs_dim1) return;

  FLOAT4 v = RI_F(data, (int2)(x, y));
  FLOAT4 clipped = clamp(v, min, max);
  WI_F(output, (int2)(x, y), clipped);
}
