__kernel void NAME(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  // FP4 va = RI_F(a, (int2)(x,y));
  // vc = OP(va, vb);

}
