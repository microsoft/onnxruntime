#pragma once

// NOTE: opnecl only check MSB...
// for MASK<n> first n value(s) is activated, select(a, b, MASK<n>) ==> MSB_SET(MASK<n>[i]) ? b[i] : a[i]
#define MASK3 (short4)(-1, -1, -1, 0)
#define MASK2 (short4)(-1, -1, 0, 0)
#define MASK1 (short4)(-1, 0, 0, 0)

#define FIRST_1(v, otherwise) select((otherwise), (v), MASK1)
#define FIRST_2(v, otherwise) select((otherwise), (v), MASK2)
#define FIRST_3(v, otherwise) select((otherwise), (v), MASK3)

// Safely gather load a 4-element vector from global memory
#define SAFE_GATHER_LDG_VEC4(v, input, base_offset, stride, remain) \
  if ((remain) > 0) {                                               \
    int i = (base_offset);                                          \
    if ((remain) >= 4) {                                            \
      (v).s0 = (input)[i];                                          \
      i += (stride);                                                \
      (v).s1 = (input)[i];                                          \
      i += (stride);                                                \
      (v).s2 = (input)[i];                                          \
      i += (stride);                                                \
      (v).s3 = (input)[i];                                          \
    } else if ((remain) == 3) {                                     \
      (v).s0 = (input)[i];                                          \
      i += (stride);                                                \
      (v).s1 = (input)[i];                                          \
      i += (stride);                                                \
      (v).s2 = (input)[i];                                          \
    } else if ((remain) == 2) {                                     \
      (v).s0 = (input)[i];                                          \
      i += (stride);                                                \
      (v).s1 = (input)[i];                                          \
    } else if ((remain) == 1) {                                     \
      (v).s0 = (input)[i];                                          \
    }                                                               \
  }

// Safely scatter store a 4-element vector to global memory
#define SAFE_SCATTER_STG_VEC4(output, base_offset, stride, remain, v) \
  if ((remain) > 0) {                                                 \
    int i = base_offset;                                              \
    if ((remain) >= 4) {                                              \
      (output)[i] = (v).s0;                                           \
      i += HW;                                                        \
      (output)[i] = (v).s1;                                           \
      i += HW;                                                        \
      (output)[i] = (v).s2;                                           \
      i += HW;                                                        \
      (output)[i] = (v).s3;                                           \
    } else if ((remain) == 3) {                                       \
      (output)[i] = (v).s0;                                           \
      i += HW;                                                        \
      (output)[i] = (v).s1;                                           \
      i += HW;                                                        \
      (output)[i] = (v).s2;                                           \
    } else if ((remain) == 2) {                                       \
      (output)[i] = (v).s0;                                           \
      i += HW;                                                        \
      (output)[i] = (v).s1;                                           \
    } else if ((remain) == 1) {                                       \
      (output)[i] = (v).s0;                                           \
    }                                                                 \
  }

// Safely load a 4-element vector from consecutive global memory
#define SAFE_VECTOR_LDG_VEC4(v, input, base_offset, remain)

// Safely store a 4-element vector to consecutive global memory
#define SAFE_VECTOR_STG_VEC4(output, base_offset, remain, v)
