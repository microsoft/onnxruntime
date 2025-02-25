#define DEFINE_GREATER_KERNEL(type, type_suffix)               \
  __kernel void greater_##type_suffix(                         \
      __global type* A,                                        \
      __global type* B,                                        \
      __global type* C,                                        \
      __global long* Ashapes,                                  \
      __global long* Bshapes,                                  \
      __global long* Cshapes,                                  \
      int maxdims) {                                           \
    int global_id = get_global_id(0);                          \
    int indices[5] = {0, 0, 0, 0, 0};                          \
    int div = global_id;                                       \
    for (int i = maxdims - 1; i >= 0; --i) {                   \
      indices[i] = div % Cshapes[i];                           \
      div /= Cshapes[i];                                       \
    }                                                          \
    int offsetA = 0;                                           \
    int offsetB = 0;                                           \
    int offsetC = 0;                                           \
    int strideA = 1;                                           \
    int strideB = 1;                                           \
    int strideC = 1;                                           \
    for (int i = maxdims - 1; i >= 0; --i) {                   \
      offsetA += (Ashapes[i] == 1 ? 0 : indices[i]) * strideA; \
      offsetB += (Bshapes[i] == 1 ? 0 : indices[i]) * strideB; \
      offsetC += indices[i] * strideC;                         \
      strideA *= Ashapes[i];                                   \
      strideB *= Bshapes[i];                                   \
      strideC *= Cshapes[i];                                   \
    }                                                          \
    C[offsetC] = (A[offsetA] > B[offsetB]) ? 1.0f : 0.0f;      \
  }

// 定义 float 类型的 greater kernel
DEFINE_GREATER_KERNEL(float, float)

// 定义 double 类型的 greater kernel
DEFINE_GREATER_KERNEL(double, double)

#ifdef USE_FP16
// 定义 half 类型的 greater kernel
#define cl_khr_fp16 1
DEFINE_GREATER_KERNEL(half, fp16)
#endif
