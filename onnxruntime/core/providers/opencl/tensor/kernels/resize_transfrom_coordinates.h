#pragma once

#define TRANS_COORD_(kind) TransformCoordinate_##kind
#define TRANS_COORD(kind) TRANS_COORD_(kind)

inline FLOAT TransformCoordinate_ASYMMETRIC(FLOAT x_resized, FLOAT x_scale_inv, FLOAT _0, FLOAT _1) {
  return x_resized * x_scale_inv;
}

#define CONST_VALUE_HALF ((FLOAT)0.5f)

inline FLOAT TransformCoordinate_HALF_PIXEL(FLOAT x_resized, FLOAT x_scale_inv, FLOAT _0, FLOAT _1) {
  return ((x_resized + CONST_VALUE_HALF) * x_scale_inv) - CONST_VALUE_HALF;
}

inline FLOAT TransformCoordinate_PYTORCH_HALF_PIXEL(FLOAT x_resized, FLOAT x_scale_inv, FLOAT length_resized, FLOAT _0) {
  FLOAT x = (x_resized + CONST_VALUE_HALF) * x_scale_inv - CONST_VALUE_HALF;
  return select((FLOAT)0.0f, x, (SELECT_PREDICATE)(length_resized > 1.0f));
}

inline FLOAT TransformCoordinate_TF_HALF_PIXEL_FOR_NN(FLOAT x_resized, FLOAT x_scale_inv, FLOAT _0, FLOAT _1) {
  return (x_resized + CONST_VALUE_HALF) * x_scale_inv;
}

inline FLOAT TransformCoordinate_ALIGN_CORNERS(FLOAT x_resized, FLOAT _0, FLOAT length_resized, FLOAT length_original) {
  return length_resized == 1 ? 0 : x_resized * (length_original - 1) / (length_resized - 1);
}
