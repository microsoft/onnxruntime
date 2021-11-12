// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

__kernel void NAME(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  FLOAT4 va = RI_F(a, (int2)(x, y));
  FLOAT4 vb = RI_F(b, (int2)(x, y));
  FLOAT4 vc;
  OP(va, vb, vc);
  WI_F(c, (int2)(x, y), vc);
}
