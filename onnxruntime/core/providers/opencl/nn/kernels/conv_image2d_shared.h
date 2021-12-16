#pragma once

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

enum ActivationKind {
  ActivationKind_None = 0,
  ActivationKind_ReLU = 1,
  ActivationKind_Clip = 5,
};

inline FLOAT4 Act(FLOAT4 v, enum ActivationKind activation_type) {
  if (activation_type == ActivationKind_None) {
    return v;
  }
  if (activation_type == ActivationKind_ReLU) {
    return fmax(v, (FLOAT4)0);
  }
  return (FLOAT4)NAN;
}

inline FLOAT4 Clip(FLOAT4 v, float clip_min, float clip_max) {
  return clamp(v, (FLOAT4)clip_min, (FLOAT4)clip_max);
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
