#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {
// These macros simplifies coding. To add a new op with following steps:
// 1. Add a new entry in CONTRIB_BINARY_OPS() list
// 2. (optional) Define templated single element operator in binary_elementwise_ops_impl.cu
// 3. (optional) Implement specialized single element operator
// 4. Add op kernel class definition in binary_elementwise_ops.h
// 5. Add op kernel registration and compute specialization in binary_elementwise_ops.cc
#define CONTRIB_BINARY_OPS() \
  CONTRIB_BINARY_OP_NAME_EXPR(AddGeluFusion, _Gelu(a + b))

// NOTE that cu files are compiled with nvcc and should not refer to any onnxruntime headers
// so struct BinaryElementwisePreparation cannot be used here
#define CONTRIB_BINARY_ELEMENTWISE_IMPL_DECLARATION(name)       \
  template <typename T>                                         \
  void Impl_##name(                                             \
      size_t output_rank_or_simple_broadcast,                   \
      const int64_t* lhs_padded_strides,                        \
      const T* lhs_data,                                        \
      const int64_t* rhs_padded_strides,                        \
      const T* rhs_data,                                        \
      const onnxruntime::cuda::fast_divmod* fdm_output_strides, \
      const onnxruntime::cuda::fast_divmod& fdm_H,              \
      const onnxruntime::cuda::fast_divmod& fdm_C,              \
      T* output_data,                                           \
      size_t count)
#define CONTRIB_BINARY_OP_NAME_EXPR(name, expr) CONTRIB_BINARY_ELEMENTWISE_IMPL_DECLARATION(name);
CONTRIB_BINARY_OPS()
#undef CONTRIB_BINARY_OP_NAME_EXPR
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
