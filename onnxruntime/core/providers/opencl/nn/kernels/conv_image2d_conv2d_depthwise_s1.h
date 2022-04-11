// Modifications Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// This file is derived from MNN (https://github.com/alibaba/MNN)

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

#pragma once

#include "./conv_image2d_shared.h"

__kernel void DepthwiseConv2DS1(
    __private const int gs_dim0,
    __private const int gs_dim1,
    __read_only image2d_t input,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __read_only image2d_t sum,
    __write_only image2d_t output,
    __private const int2 input_wh,
    __private const int2 output_wh,
    __private const int2 K,
    __private const int2 P,
    __private const int has_bias,
    __private const int has_sum,
    __private const int act_type,
    __private const float act_param0,  // only for act_type != ActivationType_identityidentity
    __private const float act_param1   // only for act_type != ActivationType_identityNone
) {
  const int output_cw_idx = get_global_id(0);
  const int output_bh_idx = get_global_id(1);
  if (output_cw_idx >= gs_dim0 || output_bh_idx >= gs_dim1) return;
  int ow4 = (output_wh.x + 3) / 4;
  const int out_channel_block_idx = output_cw_idx / ow4;
  const int out_width_block_idx = output_cw_idx % ow4;

  const int in_channel_block_idx = out_channel_block_idx;

  FLOAT4 out0 = has_bias ? RI_F(bias, (int2)(out_channel_block_idx, 0)) : (FLOAT4)0;
  FLOAT4 out1 = out0;
  FLOAT4 out2 = out0;
  FLOAT4 out3 = out0;

  const int in_width0 = mad24(out_width_block_idx, 4, -P.x);
  const int in_width1 = in_width0 + 1;
  const int in_width2 = in_width0 + 2;
  const int in_width3 = in_width0 + 3;

  int h_idx = output_bh_idx % output_wh.y - P.y;
  const int out_batch_idx =
      mul24((output_bh_idx / output_wh.y), input_wh.y);
  const int in_idx = mul24(in_channel_block_idx, input_wh.x);

  const int in_width_idx0 = select(in_idx + in_width0, -1, (in_width0 < 0 || in_width0 >= input_wh.x));
  const int in_width_idx1 = select(in_idx + in_width1, -1, (in_width1 < 0 || in_width1 >= input_wh.x));
  const int in_width_idx2 = select(in_idx + in_width2, -1, (in_width2 < 0 || in_width2 >= input_wh.x));

  FLOAT4 in0;
  FLOAT4 in1;
  FLOAT4 in2;
  FLOAT4 in3;
  for (int kh = 0; kh < K.y; kh++) {
    int in_hb_value = select(h_idx + out_batch_idx, -1, (h_idx < 0 || h_idx >= input_wh.y));
    h_idx++;
    in1 = RI_F(input, (int2)(in_width_idx0, in_hb_value));
    in2 = RI_F(input, (int2)(in_width_idx1, in_hb_value));
    in3 = RI_F(input, (int2)(in_width_idx2, in_hb_value));
    for (int kw = 0; kw < K.x; kw++) {
      int filter_idx = mad24(kh, K.x, kw);

      in0 = in1;
      in1 = in2;
      in2 = in3;

      int in_width_idx = in_width3 + kw;
      in_width_idx = select(in_idx + in_width_idx, -1, (in_width_idx < 0 || in_width_idx >= input_wh.x));
      in3 = RI_F(input, (int2)(in_width_idx, in_hb_value));

      FLOAT4 weights = RI_F(filter, (int2)(filter_idx, in_channel_block_idx));

      out0 = mad(in0, weights, out0);
      out1 = mad(in1, weights, out1);
      out2 = mad(in2, weights, out2);
      out3 = mad(in3, weights, out3);
    }
  }
  const int out_wb_idx4 = out_width_block_idx * 4;
  const int remain = output_wh.x - out_wb_idx4;
  int output_w_idx = mul24(out_channel_block_idx, output_wh.x) + out_wb_idx4;
  AddSumFusedInplace(sum, out0, out1, out2, out3, output_w_idx, output_bh_idx, remain, has_sum);
  ActivationInPlaceFloat4Vec4(out0, out1, out2, out3, act_type, CONVERT_FLOAT(act_param0), CONVERT_FLOAT(act_param1));
  SafeWriteOutput(output, out0, out1, out2, out3, output_w_idx, output_bh_idx, remain);
}
