// Modifications Copyright (c) Microsoft Corporation. All rights reserved.
// This file is derived from TNN (https://github.com/Tencent/TNN)

// Copyright (c) 2020, THL A29 Limited, a Tencent company. All rights reserved.
// Use of this source code is governed by a BSD 3-Clause license that can be
// found in the LICENSE file.

#include "./conv_image2d_shared.h"

// this kerenl only support kernel == 1x1 and padding == 0x0 and stride == 1x1
__kernel void Conv2DK1S1(
    __private const int gs_dim0,
    __private const int gs_dim1,
    __read_only image2d_t input,
    __read_only image2d_t weights,
    __read_only image2d_t bias,
    __read_only image2d_t sum,
    __write_only image2d_t output,
    __private const int2 input_wh,  // output_wh == input_wh
    __private const int input_c_blocks,
    __private const int output_w_updiv_4,
    __private const int has_bias,
    __private const int has_sum,
    __private const int act_type,
    __private const float act_param0,
    __private const float act_param1) {
  const int output_cw_idx = get_global_id(0);
  const int bh_idx = get_global_id(1);
  if (output_cw_idx >= gs_dim0 || bh_idx >= gs_dim1) return;

  const int output_c_block_idx = output_cw_idx / output_w_updiv_4;
  const int output_w_block_idx = output_cw_idx % output_w_updiv_4;

  FLOAT4 out0 = has_bias ? RI_F(bias, (int2)(output_c_block_idx, 0)) : (FLOAT4)0;
  FLOAT4 out1 = out0;
  FLOAT4 out2 = out0;
  FLOAT4 out3 = out0;

  int input_w_idx0 = output_w_block_idx << 2;
  int input_w_idx1 = input_w_idx0 + 1;
  int input_w_idx2 = input_w_idx0 + 2;
  int input_w_idx3 = input_w_idx0 + 3;

  input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= input_wh.x);
  input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= input_wh.x);
  input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= input_wh.x);
  input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= input_wh.x);

  FLOAT4 in0;
  FLOAT4 in1;
  FLOAT4 in2;
  FLOAT4 in3;
  FLOAT4 weights0;
  FLOAT4 weights1;
  FLOAT4 weights2;
  FLOAT4 weights3;
  int input_w_base = 0;
  int weights_w_base = 0;
  for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
    in0 = RI_F(input, (int2)(input_w_base + input_w_idx0, bh_idx));
    in1 = RI_F(input, (int2)(input_w_base + input_w_idx1, bh_idx));
    in2 = RI_F(input, (int2)(input_w_base + input_w_idx2, bh_idx));
    in3 = RI_F(input, (int2)(input_w_base + input_w_idx3, bh_idx));

    weights0 = RI_F(weights, (int2)(weights_w_base, output_c_block_idx));
    weights1 = RI_F(weights, (int2)(weights_w_base + 1, output_c_block_idx));
    weights2 = RI_F(weights, (int2)(weights_w_base + 2, output_c_block_idx));
    weights3 = RI_F(weights, (int2)(weights_w_base + 3, output_c_block_idx));

    CALCULATE_OUTPUT(0);
    CALCULATE_OUTPUT(1);
    CALCULATE_OUTPUT(2);
    CALCULATE_OUTPUT(3);

    input_w_base += input_wh.x;
    weights_w_base += 4;
  }

  const int out_x_base = mul24(output_c_block_idx, input_wh.x);
  int out_x_idx = output_w_block_idx << 2;

  const int remain = input_wh.x - out_x_idx;
  int output_w_idx = out_x_base + out_x_idx;
  AddSumFusedInplace(sum, out0, out1, out2, out3, output_w_idx, bh_idx, remain, has_sum);
  ActivationInPlaceFloat4Vec4(out0, out1, out2, out3, act_type, CONVERT_FLOAT(act_param0), CONVERT_FLOAT(act_param1));
  SafeWriteOutput(output, out0, out1, out2, out3, output_w_idx, bh_idx, remain);
}
