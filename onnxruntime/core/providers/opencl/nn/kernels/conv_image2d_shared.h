// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

enum MLAS_ACTIVATION_KIND {
  MlasIdentityActivation,
  MlasReluActivation,
  MlasLeakyReluActivation,
  MlasTanhActivation,
  MlasLogisticActivation,
  MlasClipActivation,
  MlasHardSigmoidActivation,
};
typedef enum MLAS_ACTIVATION_KIND ActivationKind;

#define ActivationInPlaceFloat4(out0, activation_type, firstv, secondv) \
  {                                                                     \
    if (activation_type == MlasReluActivation) {                        \
      out0 = fmax(out0, (FLOAT4)0);                                     \
    } else if (activation_type == MlasClipActivation) {                 \
      out0 = clamp(out0, (FLOAT4)firstv, (FLOAT4)secondv);              \
    } else if (activation_type == MlasLeakyReluActivation) {            \
      float f1 = 0.5 * (1.0f + firstv);                                 \
      float f2 = 0.5 * (1.0f - firstv);                                 \
      out0 = (FLOAT4)f1 * out0 + (FLOAT4)f2 * fabs(out0);               \
    } else if (activation_type == MlasTanhActivation) {                 \
      FLOAT4 v1 = native_exp(out0);                                     \
      FLOAT4 v2 = native_exp(-out0);                                    \
      out0 = native_divide(v1 - v2, v1 + v2);                           \
    } else if (activation_type == MlasLogisticActivation) {             \
      out0 = native_recip((float4)1 + native_exp(-out0));               \
    } else if (activation_type == MlasHardSigmoidActivation) {          \
      out0 = clamp(0.5f * out0 + 0.5f, 0.f, 1.0f);                      \
    }                                                                   \
}

#define ActivationInPlaceFloat4Vec4(out0, out1, out2, out3, activation_type, firstv, secondv) \
  {                                                                                           \
    if (activation_type == MlasReluActivation) {                                              \
      out0 = fmax(out0, (FLOAT4)0);                                                           \
      out1 = fmax(out1, (FLOAT4)0);                                                           \
      out2 = fmax(out2, (FLOAT4)0);                                                           \
      out3 = fmax(out3, (FLOAT4)0);                                                           \
    } else if (activation_type == MlasClipActivation) {                                       \
      out0 = clamp(out0, (FLOAT4)firstv, (FLOAT4)secondv);                                    \
      out1 = clamp(out1, (FLOAT4)firstv, (FLOAT4)secondv);                                    \
      out2 = clamp(out2, (FLOAT4)firstv, (FLOAT4)secondv);                                    \
      out3 = clamp(out3, (FLOAT4)firstv, (FLOAT4)secondv);                                    \
    } else if (activation_type == MlasLeakyReluActivation) {                                  \
      float f1 = 0.5 * (1.0f + firstv);                                                       \
      float f2 = 0.5 * (1.0f - firstv);                                                       \
      out0 = (FLOAT4)f1 * out0 + (FLOAT4)f2 * fabs(out0);                                     \
      out1 = (FLOAT4)f1 * out1 + (FLOAT4)f2 * fabs(out1);                                     \
      out2 = (FLOAT4)f1 * out2 + (FLOAT4)f2 * fabs(out2);                                     \
      out3 = (FLOAT4)f1 * out3 + (FLOAT4)f2 * fabs(out3);                                     \
    } else if (activation_type == MlasTanhActivation) {                                       \
      FLOAT4 v1 = native_exp(out0), v2 = native_exp(-out0);                                   \
      out0 = native_divide(v1 - v2, v1 + v2);                                                 \
      FLOAT4 v11 = native_exp(out1), v12 = native_exp(-out1);                                 \
      out1 = native_divide(v11 - v12, v11 + v12);                                             \
      FLOAT4 v21 = native_exp(out2), v22 = native_exp(-out2);                                 \
      out2 = native_divide(v21 - v22, v21 + v22);                                             \
      FLOAT4 v31 = native_exp(out3), v32 = native_exp(-out3);                                 \
      out3 = native_divide(v31 - v32, v31 + v32);                                             \
    } else if (activation_type == MlasLogisticActivation) {                                   \
      out0 = native_recip((float4)1 + native_exp(-out0));                                     \
      out1 = native_recip((float4)1 + native_exp(-out1));                                     \
      out2 = native_recip((float4)1 + native_exp(-out2));                                     \
      out3 = native_recip((float4)1 + native_exp(-out3));                                     \
    } else if (activation_type == MlasHardSigmoidActivation) {                                \
      out0 = clamp(0.5f * out0 + 0.5f, 0.f, 1.0f);                                            \
      out1 = clamp(0.5f * out1 + 0.5f, 0.f, 1.0f);                                            \
      out2 = clamp(0.5f * out2 + 0.5f, 0.f, 1.0f);                                            \
      out3 = clamp(0.5f * out3 + 0.5f, 0.f, 1.0f);                                            \
    }                                                                                         \
  }

#define AddSumFusedInplace(sum, out0, out1, out2, out3, output_w_idx, output_h_idx, remain, has_sum)          \
  {                                                                                                    \
    out0 += (has_sum && (remain > 0)) ? RI_F(sum, (int2)(output_w_idx, output_h_idx)) : (FLOAT4)0;     \
    out1 += (has_sum && (remain > 1)) ? RI_F(sum, (int2)(output_w_idx + 1, output_h_idx)) : (FLOAT4)0; \
    out2 += (has_sum && (remain > 2)) ? RI_F(sum, (int2)(output_w_idx + 2, output_h_idx)) : (FLOAT4)0; \
    out3 += (has_sum && (remain > 3)) ? RI_F(sum, (int2)(output_w_idx + 3, output_h_idx)) : (FLOAT4)0; \
  }

inline void SafeWriteOutput(
    __write_only image2d_t output,
    FLOAT4 out0, FLOAT4 out1, FLOAT4 out2, FLOAT4 out3,
    const int output_w_idx,
    const int output_h_idx,
    const int remain) {
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
