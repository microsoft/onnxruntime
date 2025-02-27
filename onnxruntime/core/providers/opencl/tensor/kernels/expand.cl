#define DEFINE_EXPAND_KERNEL(TYPE, OP_NAME)    \
  __kernel void OP_NAME(                       \
      __global TYPE* input,                    \
      __global TYPE* output,                   \
      __global long* input_shape,              \
      __global long* output_shape,             \
      const long Ndims) {                      \
    int gid = get_global_id(0);                \
    int output_idx[5];                         \
    int idx = gid;                             \
    for (int i = Ndims - 1; i >= 0; --i) {     \
      output_idx[i] = idx % output_shape[i];   \
      idx /= output_shape[i];                  \
    }                                          \
    int input_idx[5];                          \
    for (int i = 0; i < Ndims; ++i) {          \
      if (input_shape[i] == 1) {               \
        input_idx[i] = 0;                      \
      } else {                                 \
        input_idx[i] = output_idx[i];          \
      }                                        \
    }                                          \
    int input_linear_idx = 0;                  \
    int prod = 1;                              \
    for (int i = Ndims - 1; i >= 0; --i) {     \
      input_linear_idx += prod * input_idx[i]; \
      prod *= input_shape[i];                  \
    }                                          \
                                               \
    output[gid] = input[input_linear_idx];     \
  }

DEFINE_EXPAND_KERNEL(float, expand_float);
DEFINE_EXPAND_KERNEL(double, expand_double);
DEFINE_EXPAND_KERNEL(int, expand_int);
DEFINE_EXPAND_KERNEL(long, expand_long);
DEFINE_EXPAND_KERNEL(uint, expand_uint);
DEFINE_EXPAND_KERNEL(ulong, expand_ulong);
DEFINE_EXPAND_KERNEL(uchar, expand_uchar);
