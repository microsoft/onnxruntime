// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// launch a navie 2d grid that is width == CeilDiv(C, 4), height == N
__kernel void GlobalAveragePool(
    __private const int gs_dim0,
    __private const int gs_dim1,
    __read_only image2d_t X,
    __write_only image2d_t Y,
    __private const int2 input_wh,
    __private const float invHW_) {
#define W input_wh.s0
#define H input_wh.s1
  int c4 = get_global_id(0);
  int n = get_global_id(1);
  if (c4 >= gs_dim0 || n >= gs_dim1) return;

  int col_base = mul24(W, c4);
  int row_base = mul24(H, n);
  FLOAT4 acc = (FLOAT4)0;
  FLOAT invHW = CONVERT_FLOAT(invHW_);
  for (int h = 0; h < H; h++) {
    int row = row_base + h;
    FLOAT4 inner_acc = (FLOAT4)0;
    for (int w = 0; w < W; w++) {
      inner_acc += RI_F(X, (int2)(col_base + w, row));
    }
    acc += invHW * inner_acc;
  }

  WI_F(Y, (int2)(c4, n), acc);
#undef H
#undef W
}
