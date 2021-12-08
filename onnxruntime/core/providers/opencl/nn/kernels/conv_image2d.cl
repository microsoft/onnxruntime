// FIXME: LICENSE NOTICE:  adapted from TNN original BSD3.

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define RI_F(image, coord) read_imagef((image), (SAMPLER), (coord))
#define WI_F(image, coord, value) write_imagef((image), (coord), (value))
#define FLOAT4 float4
#define CONVERT_FLOAT4 convert_float4

#define READ_INPUT_IMAGE(i, base)                                                                         \
  int in_width_value##i = in_width##i + base;                                                             \
  in_width_value##i =                                                                                     \
      select(in_idx + in_width_value##i, -1, (in_width_value##i < 0 || in_width_value##i >= input_wh.x)); \
  in##i = RI_F(input, (int2)(in_width_value##i, in_hb_value));

#define CALCULATE_OUTPUT(i)                \
  out##i = mad(in##i.x, weights0, out##i); \
  out##i = mad(in##i.y, weights1, out##i); \
  out##i = mad(in##i.z, weights2, out##i); \
  out##i = mad(in##i.w, weights3, out##i);

enum ActivationType {
  ActivationType_None = 0,
  ActivationType_ReLU = 1,
  ActivationType_ReLU6 = 2,
};

inline FLOAT4 Act(FLOAT4 out0, enum ActivationType activation_type) {
  if (activation_type == ActivationType_ReLU) {
    return fmax(out0, (FLOAT4)0);
  } else if (activation_type == ActivationType_ReLU6) {
    return clamp(out0, (FLOAT4)0, (FLOAT4)6);
  } else {
    return out0;
  }
}

inline void SafeWriteOutput(__write_only image2d_t output, FLOAT4 out0, FLOAT4 out1, FLOAT4 out2, FLOAT4 out3, const int output_w_idx, const int output_h_idx, const int remain) {
  if (remain >= 4) {
    WI_F(output, (int2)(output_w_idx, output_h_idx), out0);
    WI_F(output, (int2)(output_w_idx + 1, output_h_idx), out1);
    WI_F(output, (int2)(output_w_idx + 2, output_h_idx), out2);
    WI_F(output, (int2)(output_w_idx + 3, output_h_idx), out3);
  } else if (remain == 3) {
    WI_F(output, (int2)(output_w_idx, output_h_idx), out0);
    WI_F(output, (int2)(output_w_idx + 1, output_h_idx), out1);
    WI_F(output, (int2)(output_w_idx + 2, output_h_idx), out2);
  } else if (remain == 2) {
    WI_F(output, (int2)(output_w_idx, output_h_idx), out0);
    WI_F(output, (int2)(output_w_idx + 1, output_h_idx), out1);
  } else if (remain == 1) {
    WI_F(output, (int2)(output_w_idx, output_h_idx), out0);
  }
}

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
    __private const int act_type) {
  const int output_cw_idx = get_global_id(0);
  const int output_bh_idx = get_global_id(1);
  if (output_cw_idx >= gs_dim0 || output_bh_idx >= gs_dim1) return;

  const int out_channel_block_idx = output_cw_idx / out_width_blocks;
  const int out_width_block_idx = output_cw_idx % out_width_blocks;

  FLOAT4 out0 = RI_F(bias, (int2)(out_channel_block_idx, 0));
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

  out0 = Act(out0, act_type);
  out1 = Act(out1, act_type);
  out2 = Act(out2, act_type);
  out3 = Act(out3, act_type);

  const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
  int out_x_idx = out_width_block_idx * 4;

  const int remain = output_wh.x - out_x_idx;
  int output_w_idx = out_x_base + out_x_idx;
  SafeWriteOutput(output, out0, out1, out2, out3, output_w_idx, output_bh_idx, remain);
}

__kernel void DepthwiseConv2D(
    __private const int gs_dim0,
    __private const int gs_dim1,
    __read_only image2d_t input,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output,
    __private const int2 input_wh,
    __private const int2 output_wh,
    __private const int2 kernel_wh,
    __private const int2 stride_wh,
    __private const int2 padding_wh,
    __private const int2 dilation_wh,
    __private const int act_type) {
  const int output_cw_idx = get_global_id(0);
  const int output_bh_idx = get_global_id(1);
  if (output_cw_idx >= gs_dim0 || output_bh_idx >= gs_dim1) return;

  int out_width_blocks = (output_wh.x + 3) / 4;
  const int out_channel_block_idx = output_cw_idx / out_width_blocks;
  const int out_width_block_idx = output_cw_idx % out_width_blocks;

  const int in_channel_block_idx = out_channel_block_idx;

  FLOAT4 out0 = RI_F(bias, (int2)(out_channel_block_idx, 0));
  FLOAT4 out1 = out0;
  FLOAT4 out2 = out0;
  FLOAT4 out3 = out0;

  const int in_width0 = mad24(out_width_block_idx, stride_wh.x * 4, -padding_wh.x);
  const int in_width1 = in_width0 + stride_wh.x;
  const int in_width2 = in_width1 + stride_wh.x;
  const int in_width3 = in_width2 + stride_wh.x;
  int h_idx = mad24(output_bh_idx % output_wh.y, stride_wh.y, -padding_wh.y);

  const int out_b_idx = mul24((output_bh_idx / output_wh.y), input_wh.y);

  const int in_idx = mul24(in_channel_block_idx, input_wh.x);
  for (int kh = 0; kh < kernel_wh.y; kh++) {
    int in_hb_value = select(h_idx + out_b_idx, -1, (h_idx < 0 || h_idx >= input_wh.y));
    h_idx += dilation_wh.y;
    for (int kw = 0; kw < kernel_wh.x; kw++) {
      int filter_idx = mad24(kh, kernel_wh.x, kw);
      FLOAT4 in0;
      FLOAT4 in1;
      FLOAT4 in2;
      FLOAT4 in3;
      int input_w_base = mul24(kw, dilation_wh.x);

      READ_INPUT_IMAGE(0, input_w_base);
      READ_INPUT_IMAGE(1, input_w_base);
      READ_INPUT_IMAGE(2, input_w_base);
      READ_INPUT_IMAGE(3, input_w_base);

      FLOAT4 weights = RI_F(filter, (int2)(filter_idx, in_channel_block_idx));

      out0 = mad(in0, weights, out0);
      out1 = mad(in1, weights, out1);
      out2 = mad(in2, weights, out2);
      out3 = mad(in3, weights, out3);
    }
  }

  out0 = Act(out0, act_type);
  out1 = Act(out1, act_type);
  out2 = Act(out2, act_type);
  out3 = Act(out3, act_type);

  const int out_wb_idx4 = out_width_block_idx * 4;
  const int remain = output_wh.x - out_wb_idx4;
  int output_w_idx = mad24(out_channel_block_idx, output_wh.x, out_wb_idx4);
  SafeWriteOutput(output, out0, out1, out2, out3, output_w_idx, output_bh_idx, remain);
}
