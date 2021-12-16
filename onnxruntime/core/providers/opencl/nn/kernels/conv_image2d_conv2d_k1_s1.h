#include "./conv_image2d_shared.h"

// this kerenl only support kernel == 1x1 and stride == 1x1
__kernel void Conv2D_K1_S1(
    __private const int gs_dim0,
    __private const int gs_dim1,
    __read_only image2d_t input,
    __read_only image2d_t weights,
    __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 wh,
    __private const int input_c_blocks,
    __private const int output_w_updiv_4,
    __private const int act_type,
    __private const int act_param0,
    __private const int act_param1) {
  const int output_cw_idx = get_global_id(0);
  const int bh_idx = get_global_id(1);
  if (output_cw_idx >= gs_dim0 || bh_idx >= gs_dim1) return;

  const int output_c_block_idx = output_cw_idx / output_w_updiv_4;
  const int output_w_block_idx = output_cw_idx % output_w_updiv_4;

  FLOAT4 out0 = RI_F(bias, (int2)(output_c_block_idx, 0));
  FLOAT4 out1 = out0;
  FLOAT4 out2 = out0;
  FLOAT4 out3 = out0;

  int input_w_idx0 = output_w_block_idx << 2;
  int input_w_idx1 = input_w_idx0 + 1;
  int input_w_idx2 = input_w_idx0 + 2;
  int input_w_idx3 = input_w_idx0 + 3;

  input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= wh.x);
  input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= wh.x);
  input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= wh.x);
  input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= wh.x);

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

    input_w_base += wh.x;
    weights_w_base += 4;
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

  const int out_x_base = mul24(output_c_block_idx, wh.x);
  int out_x_idx = output_w_block_idx << 2;

  const int remain = wh.x - out_x_idx;
  int output_w_idx = out_x_base + out_x_idx;
  SafeWriteOutput(output, out0, out1, out2, out3, output_w_idx, bh_idx, remain);
}
