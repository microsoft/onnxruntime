// Modifications Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Copyright (c) 2018, Alibaba Group Holding Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "./conv_image2d_shared.h"

// this kerenl only support kernel == 1x1 and padding == 0x0
__kernel void Conv2DK1(
    __private const int gs_dim0,
    __private const int gs_dim1,
    __read_only image2d_t input,
    __read_only image2d_t weights,
    __read_only image2d_t bias,
    __read_only image2d_t sum,
    __write_only image2d_t output,
    __private const int2 input_wh,
    __private const int input_c_blocks,
    __private const int2 output_wh,
    __private const int2 stride_wh,
    __private const int output_w_updiv_4,
    __private const int has_bias,
    __private const int has_sum,
    __private const int act_type,
    __private const float act_param0,
    __private const float act_param1) {
  const int output_cw_idx = get_global_id(0);
  const int output_bh_idx = get_global_id(1);
  if (output_cw_idx >= gs_dim0 || output_bh_idx >= gs_dim1) return;

  const int output_c_block_idx = output_cw_idx / output_w_updiv_4;
  const int output_w_block_idx = output_cw_idx % output_w_updiv_4;

  FLOAT4 out0 = has_bias ? RI_F(bias, (int2)(output_c_block_idx, 0)) : (FLOAT4)0;
  FLOAT4 out1 = out0;
  FLOAT4 out2 = out0;
  FLOAT4 out3 = out0;

  int input_w_idx0 = mul24(output_w_block_idx, stride_wh.x << 2);
  int input_w_idx1 = input_w_idx0 + stride_wh.x;
  int input_w_idx2 = input_w_idx1 + stride_wh.x;
  int input_w_idx3 = input_w_idx2 + stride_wh.x;

  input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= input_wh.x);
  input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= input_wh.x);
  input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= input_wh.x);
  input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= input_wh.x);

  int b_idx = output_bh_idx / output_wh.y;
  int input_bh_idx = mad24(output_bh_idx % output_wh.y, stride_wh.y, b_idx * input_wh.y);

  FLOAT4 in0;
  FLOAT4 in1;
  FLOAT4 in2;
  FLOAT4 in3;
  FLOAT4 weights0;
  FLOAT4 weights1;
  FLOAT4 weights2;
  FLOAT4 weights3;
  for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
    int input_w_base = input_c_block_idx * input_wh.x;
    int weights_w_base = input_c_block_idx << 2;

    in0 = RI_F(input, (int2)(input_w_base + input_w_idx0, input_bh_idx));
    in1 = RI_F(input, (int2)(input_w_base + input_w_idx1, input_bh_idx));
    in2 = RI_F(input, (int2)(input_w_base + input_w_idx2, input_bh_idx));
    in3 = RI_F(input, (int2)(input_w_base + input_w_idx3, input_bh_idx));

    weights0 = RI_F(weights, (int2)(weights_w_base, output_c_block_idx));
    weights1 = RI_F(weights, (int2)(weights_w_base + 1, output_c_block_idx));
    weights2 = RI_F(weights, (int2)(weights_w_base + 2, output_c_block_idx));
    weights3 = RI_F(weights, (int2)(weights_w_base + 3, output_c_block_idx));

    CALCULATE_OUTPUT(0);
    CALCULATE_OUTPUT(1);
    CALCULATE_OUTPUT(2);
    CALCULATE_OUTPUT(3);
  }
  const int out_x_base = mul24(output_c_block_idx, output_wh.x);
  int out_x_idx = output_w_block_idx << 2;

  const int remain = output_wh.x - out_x_idx;
  int output_w_idx = out_x_base + out_x_idx;
  //conv_add_activation
  AddSumFusedInplace(sum, out0, out1, out2, out3, output_w_idx, output_bh_idx, remain, has_sum);
  ActivationInPlaceFloat4Vec4(out0, out1, out2, out3, act_type, (act_param0), CONVERT_FLOAT(act_param1));
  SafeWriteOutput(output, out0, out1, out2, out3, output_w_idx, output_bh_idx, remain);
}
