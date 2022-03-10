// FIXME: LICENSE NOTICE:  adapted from TNN original BSD3.
#pragma once

#include "./conv_image2d_shared.h"

__kernel void Conv2D(
    __private const int gs_dim0,
    __private const int gs_dim1,
    __read_only image2d_t input,
    __read_only image2d_t weights,
    __read_only image2d_t bias,
    __write_only image2d_t output,
    __private const int2 input_wh,
    __private const int in_channel_block_length,
    __private const int2 output_wh,
    __private const int2 kernel_wh,
    __private const int2 stride_wh,
    __private const int2 padding_wh,
    __private const int2 dilation_wh,
    __private const int out_width_blocks,
    __private const int has_bias,
    __private const int act_type,
    __private const float act_param0,  // only for act_type != ActivationType_None
    __private const float act_param1   // only for act_type != ActivationType_None
) {
  const int output_cw_idx = get_global_id(0);
  const int output_bh_idx = get_global_id(1);
  if (output_cw_idx >= gs_dim0 || output_bh_idx >= gs_dim1) return;

  const int out_channel_block_idx = output_cw_idx / out_width_blocks;
  const int out_width_block_idx = output_cw_idx % out_width_blocks;

  FLOAT4 out0 = has_bias ? RI_F(bias, (int2)(out_channel_block_idx, 0)) : (FLOAT4)0;
  FLOAT4 out1 = out0;
  FLOAT4 out2 = out0;
  FLOAT4 out3 = out0;

  int in_width0 = mad24(out_width_block_idx, stride_wh.x * 4, -padding_wh.x);
  int in_width1 = in_width0 + stride_wh.x;
  int in_width2 = in_width1 + stride_wh.x;
  int in_width3 = in_width2 + stride_wh.x;

  const int height_start = mad24((output_bh_idx % output_wh.y), stride_wh.y, -padding_wh.y);
  int in_height_start = mad24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0), dilation_wh.y, height_start);
  int in_height_end = min(mad24(kernel_wh.y, dilation_wh.y, height_start), input_wh.y);

  const int batch_idx = mul24((output_bh_idx / output_wh.y), input_wh.y);
  const int weights_h_idx = mul24(out_channel_block_idx, mul24(kernel_wh.x, kernel_wh.y)) +
                            mul24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0), kernel_wh.x);

  FLOAT4 in0;
  FLOAT4 in1;
  FLOAT4 in2;
  FLOAT4 in3;
  FLOAT4 weights0;
  FLOAT4 weights1;
  FLOAT4 weights2;
  FLOAT4 weights3;
  for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
    const int in_idx = mul24(input_c_block_idx, input_wh.x);
    int weights_x_idx = input_c_block_idx << 2;
    int weights_y_idx = weights_h_idx;
    for (int iy = in_height_start; iy < in_height_end; iy += dilation_wh.y) {
      int in_hb_value = iy + batch_idx;
      for (int w = 0; w < kernel_wh.x; w++) {
        int input_w_base = mul24(w, dilation_wh.x);
        READ_INPUT_IMAGE(0, input_w_base);
        READ_INPUT_IMAGE(1, input_w_base);
        READ_INPUT_IMAGE(2, input_w_base);
        READ_INPUT_IMAGE(3, input_w_base);

        weights0 = RI_F(weights, (int2)(weights_x_idx, weights_y_idx));
        weights1 = RI_F(weights, (int2)(weights_x_idx + 1, weights_y_idx));
        weights2 = RI_F(weights, (int2)(weights_x_idx + 2, weights_y_idx));
        weights3 = RI_F(weights, (int2)(weights_x_idx + 3, weights_y_idx));
        weights_y_idx += 1;

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);
      }
    }
  }

  if (act_type == ActivationKind_Clip) {
    out0 = Clip(out0, act_param0, act_param1);
    out1 = Clip(out1, act_param0, act_param1);
    out2 = Clip(out2, act_param0, act_param1);
    out3 = Clip(out3, act_param0, act_param1);
  } else {
    out0 = Act(out0, act_type);
    out1 = Act(out1, act_type);
    out2 = Act(out2, act_type);
    out3 = Act(out3, act_type);
  }

  const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
  int out_x_idx = out_width_block_idx * 4;

  const int remain = output_wh.x - out_x_idx;
  int output_w_idx = out_x_base + out_x_idx;
  SafeWriteOutput(output, out0, out1, out2, out3, output_w_idx, output_bh_idx, remain);
}
