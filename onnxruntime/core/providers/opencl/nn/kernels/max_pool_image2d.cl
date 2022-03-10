// FIXME: LICENSE NOTICE:  adapted from TNN original BSD3.

__kernel void MaxPool(
    __private const int3 global_size,
    __read_only image2d_t input,
    __write_only image2d_t output,
    __private const int2 input_wh,
    __private const int output_height,
    __private const int2 K,
    __private const int2 S,
    __private const int2 P) {
  const int output_c_idx = get_global_id(0);
  const int output_w_idx = get_global_id(1);
  const int output_bh_idx = get_global_id(2);

  if (output_c_idx >= global_size.x || output_w_idx >= global_size.y || output_bh_idx >= global_size.z) return;
  const int output_width = global_size.y;

  const int output_b_idx = output_bh_idx / output_height;
  const int output_h_idx = output_bh_idx - mul24(output_b_idx, output_height);
  const int input_start = mul24(output_b_idx, input_wh.y);
  const int input_h_start = mad24(output_h_idx, S.y, -P.y);
  const int input_w_start = mad24(output_w_idx, S.x, -P.x);
  const int input_c_start = mul24(output_c_idx, input_wh.x);

  FLOAT4 output_result = (FLOAT4)(-FLT_MAX);
  for (int height = 0; height < K.y; height++) {
    int input_h_idx = input_h_start + height;
    input_h_idx =
        select(input_start + input_h_idx, -1, (input_h_idx < 0 || input_h_idx >= input_wh.y));
    if (input_h_idx != -1) {
      for (int width = 0; width < K.x; width++) {
        int input_w_idx = input_w_start + width;
        input_w_idx = select(input_c_start + input_w_idx, -1,
                                 (input_w_idx < 0 || input_w_idx >= input_wh.x));

        if (input_w_idx != -1) {
          FLOAT4 input_data = RI_F(input, (int2)(input_w_idx, input_h_idx));
          output_result = fmax(output_result, input_data);
        }
      }
    }
  }

  const int output_cw_idx = mad24(output_c_idx, output_width, output_w_idx);
  WI_F(output, (int2)(output_cw_idx, output_bh_idx), output_result);
}
