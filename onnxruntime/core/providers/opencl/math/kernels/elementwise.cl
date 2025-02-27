// Simple operations macros
#define ADD_OP(a, b) ((a) + (b))
#define SUB_OP(a, b) ((a) - (b))
#define MUL_OP(a, b) ((a) * (b))
#define DIV_OP(a, b) ((a) / (b))
#define POW_OP(a, b) (pow((a), (b)))

// Kernel definition macro with broadcasting support for up to 5 dimensions
#define DEFINE_OP_KERNEL(type, type_suffix, op_name, operation)           \
  __kernel void op_name##_##type_suffix(                                  \
      __global type* A,                                                   \
      __global type* B,                                                   \
      __global type* C,                                                   \
      __global long* Ashapes,                                             \
      __global long* Bshapes,                                             \
      __global long* Cshapes,                                             \
      int Adims,                                                          \
      int Bdims,                                                          \
      int Cdims) {                                                        \
    int global_id = get_global_id(0);                                     \
    int indices[5] = {0, 0, 0, 0, 0};                                     \
    int div = global_id;                                                  \
                                                                          \
    /* Calculate the index for each dimension in C (output tensor) */     \
    for (int i = Cdims - 1; i >= 0; --i) {                                \
      indices[i] = div % Cshapes[i];                                      \
      div /= Cshapes[i];                                                  \
    }                                                                     \
                                                                          \
    /* Map the broadcasted indices to A and B */                          \
    int A_idx = 0, B_idx = 0;                                             \
    int A_stride = 1, B_stride = 1;                                       \
                                                                          \
    /* Iterate from the highest dimension down to the lowest */           \
    for (int i = Cdims - 1; i >= 0; --i) {                                \
      int ashape_id = Adims - Cdims + i;                                  \
      int A_dim_size = (ashape_id < 0) ? 1 : Ashapes[ashape_id];          \
      int bshape_id = Bdims - Cdims + i;                                  \
      int B_dim_size = (bshape_id < 0) ? 1 : Bshapes[bshape_id];          \
      int idx = indices[i];                                               \
                                                                          \
      /* Adjust for broadcasting: only add index if dimension size > 1 */ \
      if (A_dim_size > 1) {                                               \
        A_idx += idx * A_stride;                                          \
      }                                                                   \
      if (B_dim_size > 1) {                                               \
        B_idx += idx * B_stride;                                          \
      }                                                                   \
                                                                          \
      A_stride *= A_dim_size;                                             \
      B_stride *= B_dim_size;                                             \
    }                                                                     \
                                                                          \
    /* Perform the operation with broadcasting */                         \
    C[global_id] = operation(A[A_idx], B[B_idx]);                         \
  }

// Define float type kernel for add operation
DEFINE_OP_KERNEL(float, float, add, ADD_OP)

DEFINE_OP_KERNEL(float, float, sub, SUB_OP)
DEFINE_OP_KERNEL(float, float, mul, MUL_OP)
DEFINE_OP_KERNEL(float, float, div, DIV_OP)
DEFINE_OP_KERNEL(float, float, pow, POW_OP)

// double type kernels
DEFINE_OP_KERNEL(double, double, add, ADD_OP)
DEFINE_OP_KERNEL(double, double, sub, SUB_OP)
DEFINE_OP_KERNEL(double, double, mul, MUL_OP)
DEFINE_OP_KERNEL(double, double, div, DIV_OP)
DEFINE_OP_KERNEL(double, double, pow, POW_OP)

// int type kernels
DEFINE_OP_KERNEL(int, int, add, ADD_OP)
DEFINE_OP_KERNEL(int, int, sub, SUB_OP)
DEFINE_OP_KERNEL(int, int, mul, MUL_OP)
DEFINE_OP_KERNEL(int, int, div, DIV_OP)
// DEFINE_OP_KERNEL(int, int, pow, POW_OP)

// long type kernels
DEFINE_OP_KERNEL(long, long, add, ADD_OP)
DEFINE_OP_KERNEL(long, long, sub, SUB_OP)
DEFINE_OP_KERNEL(long, long, mul, MUL_OP)
DEFINE_OP_KERNEL(long, long, div, DIV_OP)
// DEFINE_OP_KERNEL(long, long, pow, POW_OP)

#ifdef USE_FP16
#define cl_khr_fp16 1
DEFINE_OP_KERNEL(half, fp16, add, ADD_OP)
DEFINE_OP_KERNEL(half, fp16, sub, SUB_OP)
DEFINE_OP_KERNEL(half, fp16, mul, MUL_OP)
DEFINE_OP_KERNEL(half, fp16, div, DIV_OP)
DEFINE_OP_KERNEL(half, fp16, pow, POW_OP)
#endif

// Macros for comparison operations
#define GREATER_OP(a, b) ((a) > (b) ? 1 : 0)
#define EQUAL_OP(a, b) ((a) == (b) ? 1 : 0)

#define DEFINE_LOGICOP_KERNEL(type, type_suffix, op_name, operation)      \
  __kernel void op_name##_##type_suffix(                                  \
      __global type* A,                                                   \
      __global type* B,                                                   \
      __global uchar* C,                                                  \
      __global long* Ashapes,                                             \
      __global long* Bshapes,                                             \
      __global long* Cshapes,                                             \
      int Adims,                                                          \
      int Bdims,                                                          \
      int Cdims) {                                                        \
    int global_id = get_global_id(0);                                     \
    int indices[5] = {0, 0, 0, 0, 0};                                     \
    int div = global_id;                                                  \
                                                                          \
    /* Calculate the index for each dimension in C (output tensor) */     \
    for (int i = Cdims - 1; i >= 0; --i) {                                \
      indices[i] = div % Cshapes[i];                                      \
      div /= Cshapes[i];                                                  \
    }                                                                     \
                                                                          \
    /* Map the broadcasted indices to A and B */                          \
    int A_idx = 0, B_idx = 0;                                             \
    int A_stride = 1, B_stride = 1;                                       \
                                                                          \
    /* Iterate from the highest dimension down to the lowest */           \
    for (int i = Cdims - 1; i >= 0; --i) {                                \
      int ashape_id = Adims - Cdims + i;                                  \
      int A_dim_size = (ashape_id < 0) ? 1 : Ashapes[ashape_id];          \
      int bshape_id = Bdims - Cdims + i;                                  \
      int B_dim_size = (bshape_id < 0) ? 1 : Bshapes[bshape_id];          \
      int idx = indices[i];                                               \
                                                                          \
      /* Adjust for broadcasting: only add index if dimension size > 1 */ \
      if (A_dim_size > 1) {                                               \
        A_idx += idx * A_stride;                                          \
      }                                                                   \
      if (B_dim_size > 1) {                                               \
        B_idx += idx * B_stride;                                          \
      }                                                                   \
                                                                          \
      A_stride *= A_dim_size;                                             \
      B_stride *= B_dim_size;                                             \
    }                                                                     \
                                                                          \
    /* Perform the operation with broadcasting */                         \
    C[global_id] = operation(A[A_idx], B[B_idx]);                         \
  }

// float type kernels for Greater and Equal
DEFINE_LOGICOP_KERNEL(float, float, greater, GREATER_OP)
DEFINE_LOGICOP_KERNEL(float, float, equal, EQUAL_OP)

// double type kernels for Greater and Equal
DEFINE_LOGICOP_KERNEL(double, double, greater, GREATER_OP)
DEFINE_LOGICOP_KERNEL(double, double, equal, EQUAL_OP)

// int type kernels for Greater and Equal
DEFINE_LOGICOP_KERNEL(int, int, greater, GREATER_OP)
DEFINE_LOGICOP_KERNEL(int, int, equal, EQUAL_OP)

// long type kernels for Greater and Equal
DEFINE_LOGICOP_KERNEL(long, long, greater, GREATER_OP)
DEFINE_LOGICOP_KERNEL(long, long, equal, EQUAL_OP)

DEFINE_LOGICOP_KERNEL(uchar, bool, equal, EQUAL_OP)
