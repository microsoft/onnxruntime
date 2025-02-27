
#define DEFINE_NEG_KERNEL(type, type_suffix) \
  __kernel void neg_##type_suffix(           \
      __global type* input,                  \
      __global type* output) {               \
    int gid = get_global_id(0);              \
    output[gid] = -input[gid];               \
  }

DEFINE_NEG_KERNEL(float, float)
DEFINE_NEG_KERNEL(double, double)
#ifdef USE_FP16
#define cl_khr_fp16 1
DEFINE_NEG_KERNEL(half, fp16)
#endif
