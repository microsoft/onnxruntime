__kernel void ReLU(
    const int gs_dim0,
    const int gs_dim1,
    __read_only image2d_t data,
    __write_only image2d_t output,
    __private const float alpha) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= gs_dim0 || y >= gs_dim1) return;
  FLOAT4 v = RI_F(data, (int2)(x, y));
  FLOAT4 zero = (FLOAT4)0;
  if (alpha != 0) {
    // v[i] = v[i] > zero[i] ? v[i] : max(...)[i]
    v = select(max(zero, v), v, v > zero);
  } else {
    v = max(zero, v);
  }
  WI_F(output, (int2)(x, y), v);
}
