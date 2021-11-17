__kernel void NAME(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  float4 va = read_imagef(a, (int2)(x, y));
  float4 vb = read_imagef(b, (int2)(x, y));
  float4 vc;
  OP(va, vb, vc);
  write_imagef(c, (int2)(x,y), vc);
}
