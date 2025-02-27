#define DEFINE_OP_KERNEL(TYPE, OP_NAME)                                 \
  __kernel void OP_NAME(                                                \
      __global uchar* condition,                                        \
      __global TYPE* x,                                                 \
      __global TYPE* y,                                                 \
      __global TYPE* output,                                            \
      __global long* condition_shape,                                   \
      __global long* condition_stride,                                  \
      __global long* x_shape,                                           \
      __global long* x_stride,                                          \
      __global long* y_shape,                                           \
      __global long* y_stride,                                          \
      __global long* output_shape,                                      \
      __global long* output_stride,                                     \
      long Ndims) {                                                     \
    long global_id = get_global_id(0);                                  \
    long idx[5];                                                        \
    long temp = global_id;                                              \
    for (long i = 0; i < Ndims; i++) {                                  \
      idx[i] = temp / output_stride[i];                                 \
      temp %= output_stride[i];                                         \
    }                                                                   \
    long condition_idx = 0;                                             \
    long x_idx = 0;                                                     \
    long y_idx = 0;                                                     \
    for (long i = 0; i < Ndims; i++) {                                  \
      long cond_dim = condition_shape[i];                               \
      long x_dim = x_shape[i];                                          \
      long y_dim = y_shape[i];                                          \
      long cond_idx = (cond_dim == 1) ? 0 : idx[i];                     \
      long x_idx_dim = (x_dim == 1) ? 0 : idx[i];                       \
      long y_idx_dim = (y_dim == 1) ? 0 : idx[i];                       \
      condition_idx += cond_idx * condition_stride[i];                  \
      x_idx += x_idx_dim * x_stride[i];                                 \
      y_idx += y_idx_dim * y_stride[i];                                 \
    }                                                                   \
    output[global_id] = condition[condition_idx] ? x[x_idx] : y[y_idx]; \
  }

DEFINE_OP_KERNEL(float, where_float)

DEFINE_OP_KERNEL(int, where_int)

DEFINE_OP_KERNEL(long, where_long)

DEFINE_OP_KERNEL(double, where_double)
