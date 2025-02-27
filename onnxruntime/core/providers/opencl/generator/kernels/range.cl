#define DEFINE_RANGE_KERNEL(T)                                         \
  __kernel void Range_##T(__global T* out, T start, T delta, long n) { \
    int gid = get_global_id(0);                                        \
    out[gid] = start + gid * delta;                                    \
  }
DEFINE_RANGE_KERNEL(int)
DEFINE_RANGE_KERNEL(float)
DEFINE_RANGE_KERNEL(long)
DEFINE_RANGE_KERNEL(double)
