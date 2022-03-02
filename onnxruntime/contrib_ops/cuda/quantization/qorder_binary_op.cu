#include "qorder_binary_op.cuh"

#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"

using onnxruntime::cuda::BinaryElementWiseImpl;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define QORDERED_BINARY_THREE_SCALE_IMPL(name)                               \
  QORDERED_BINARY_THREE_SCALE_DECLARATION(name) {                            \
    BinaryElementWiseImpl(stream,                                            \
                          output_rank_or_simple_broadcast,                   \
                          lhs_padded_strides,                                \
                          lhs_data,                                          \
                          rhs_padded_strides,                                \
                          rhs_data,                                          \
                          fdm_output_strides,                                \
                          fdm_H,                                             \
                          fdm_C,                                             \
                          output_data,                                       \
                          QORDERED_OP_##name(lhs_scale, rhs_scale, y_scale), \
                          count);                                            \
  }

#define QORDERED_BINARY_TWO_SCALE_IMPL(name)                        \
  QORDERED_BINARY_TWO_SCALE_DECLARATION(name) {                     \
    BinaryElementWiseImpl(stream,                                   \
                          output_rank_or_simple_broadcast,          \
                          lhs_padded_strides,                       \
                          lhs_data,                                 \
                          rhs_padded_strides,                       \
                          rhs_data,                                 \
                          fdm_output_strides,                       \
                          fdm_H,                                    \
                          fdm_C,                                    \
                          output_data,                              \
                          QORDERED_OP_##name(lhs_scale, rhs_scale), \
                          count);                                   \
  }

struct QORDERED_OP_Add {
  float scaleA_;
  float scaleB_;
  QORDERED_OP_Add(float scaleA, float scaleB)
      : scaleA_(scaleA), scaleB_(scaleB) {
  }
  __device__ __inline__ int8_t operator()(int8_t a, int8_t b) const {
    float v = scaleA_ * a + scaleB_ * b;
    v = fmaxf(fminf(v, 127.0f), -128.0f);
    return static_cast<int8_t>(lrintf(v));
  }
};

QORDERED_BINARY_TWO_SCALE_IMPL(Add);

// constants for approximating the normal cdf
constexpr float A = 0.5f;
constexpr float B = 0.7978845608028654f;    // sqrt(2.0/M_PI)
constexpr float C = 0.035677408136300125f;  // 0.044715 * sqrt(2.0/M_PI)

struct QORDERED_OP_BiasGelu {
  float scaleA_;
  float scaleB_;
  float scaleY_A_;

  QORDERED_OP_BiasGelu(float scaleA, float scaleB, float scaleY)
      : scaleA_(scaleA), scaleB_(scaleB), scaleY_A_(scaleY * A) {
  }

  __device__ __inline__ int8_t operator()(int8_t a, int8_t b) const {
    float x = scaleA_ * a + scaleB_ * b;
    float y = x * (scaleY_A_ + scaleY_A_ * tanhf(x * (C * x * x + B)));
    y = fmaxf(fminf(y, 127.0f), -128.0f);
    return static_cast<int8_t>(lrintf(y));
  }
};

QORDERED_BINARY_THREE_SCALE_IMPL(BiasGelu);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
