// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define MODE_HALF_PIXEL 0
#define MODE_ASYMMETRIC 1
#define MODE_PYTORCH_HALF_PIXEL 2
#define MODE_TF_HALF_PIXEL_FOR_NN 3
#define MODE_ALIGN_CORNERS 4
// #define MODE_TF_CROP_AND_RESIZE 5 // UNSUPPORT

#define TRANS_COORDS(mode,                                                                          \
                     xout, yout,                                                                    \
                     x_resized, y_resized,                                                          \
                     x_scale_inv, y_scale_inv,                                                      \
                     x_length_resized, y_length_resized,                                            \
                     x_length_original, y_length_original)                                          \
  if ((mode) == MODE_HALF_PIXEL) {                                                                  \
    xout = TransformCoordinate_HALF_PIXEL((x_resized), (x_scale_inv));                              \
    yout = TransformCoordinate_HALF_PIXEL((y_resized), (y_scale_inv));                              \
  } else if ((mode) == MODE_ASYMMETRIC) {                                                           \
    xout = TransformCoordinate_ASYMMETRIC((x_resized), (x_scale_inv));                              \
    yout = TransformCoordinate_ASYMMETRIC((y_resized), (y_scale_inv));                              \
  } else if ((mode) == MODE_PYTORCH_HALF_PIXEL) {                                                   \
    xout = TransformCoordinate_PYTORCH_HALF_PIXEL((x_resized), (x_scale_inv), (x_length_resized));  \
    yout = TransformCoordinate_PYTORCH_HALF_PIXEL((y_resized), (y_scale_inv), (y_length_resized));  \
  } else if ((mode) == MODE_TF_HALF_PIXEL_FOR_NN) {                                                 \
    xout = TransformCoordinate_TF_HALF_PIXEL_FOR_NN((x_resized), (x_scale_inv));                    \
    yout = TransformCoordinate_TF_HALF_PIXEL_FOR_NN((y_resized), (y_scale_inv));                    \
  } else if ((mode) == MODE_ALIGN_CORNERS) {                                                        \
    xout = TransformCoordinate_ALIGN_CORNERS((x_resized), (x_length_resized), (x_length_original)); \
    yout = TransformCoordinate_ALIGN_CORNERS((y_resized), (y_length_resized), (y_length_original)); \
  }

#define CONST_VALUE_HALF ((FLOAT)0.5f)

inline FLOAT TransformCoordinate_HALF_PIXEL(FLOAT x_resized, FLOAT x_scale_inv) {
  return ((x_resized + CONST_VALUE_HALF) * x_scale_inv) - CONST_VALUE_HALF;
}

inline FLOAT TransformCoordinate_ASYMMETRIC(FLOAT x_resized, FLOAT x_scale_inv) {
  return x_resized * x_scale_inv;
}

inline FLOAT TransformCoordinate_PYTORCH_HALF_PIXEL(FLOAT x_resized, FLOAT x_scale_inv, FLOAT length_resized) {
  FLOAT x = (x_resized + CONST_VALUE_HALF) * x_scale_inv - CONST_VALUE_HALF;
  return select((FLOAT)0.0f, x, (SELECT_PREDICATE)(length_resized > 1.0f));
}

inline FLOAT TransformCoordinate_TF_HALF_PIXEL_FOR_NN(FLOAT x_resized, FLOAT x_scale_inv) {
  return (x_resized + CONST_VALUE_HALF) * x_scale_inv;
}

inline FLOAT TransformCoordinate_ALIGN_CORNERS(FLOAT x_resized, FLOAT length_resized, FLOAT length_original) {
  return length_resized == 1 ? 0 : x_resized * (length_original - 1) / (length_resized - 1);
}
