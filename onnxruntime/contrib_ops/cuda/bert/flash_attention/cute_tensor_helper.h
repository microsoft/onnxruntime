#pragma once

// #include <cute/tensor.hpp>
// #include <cutlass/half.h>
// #include <cutlass/bfloat16.h>
// #include <stdio.h>

// __device__ inline float to_float(float x) { return x; }

// __device__ inline float to_float(cutlass::half_t x) {
// #if __CUDA_ARCH__ >= 530
//   return __half2float(static_cast<__half>(x));
// #else
//   return static_cast<float>(x);
// #endif
// }

// __device__ inline float to_float(cutlass::bfloat16_t x) {
//   return static_cast<float>(x);
// }

// namespace cute {

// // Prints 2D tensor to buffer (each row ends with \n, buffer is null-terminated)
// template <typename Tensor>
// __device__ void print_tensor_to_buffer(const Tensor& t,
//                                        char* buffer,
//                                        int buffer_stride,
//                                        int max_rows,
//                                        int max_cols) {
//   int row_limit = size<0>(t);
//   int col_limit = size<1>(t);

//   for (int i = 0; i < row_limit && i < max_rows; ++i) {
//     for (int j = 0; j < col_limit && j < max_cols; ++j) {
//       int offset = i * buffer_stride + j * 12;  // each float: up to 11 chars + space
//       float val = to_float(t(i, j));
//       sprintf(&buffer[offset], "%8.4f ", val);
//     }
//     int newline_offset = i * buffer_stride + col_limit * 12;
//     buffer[newline_offset] = '\n';
//   }

//   // Null terminate after last row
//   int end_offset = row_limit * buffer_stride;
//   buffer[end_offset] = '\0';
// }

// // Prints 1D tensor to buffer
// template <typename Tensor>
// __device__ void print_1d_tensor_to_buffer(const Tensor& t,
//                                           char* buffer,
//                                           int buffer_stride) {
//   int len = size(t);
//   for (int i = 0; i < len; ++i) {
//     int offset = i * 12;
//     float val = to_float(t(i));
//     sprintf(&buffer[offset], "%8.4f ", val);
//   }
//   buffer[len * 12] = '\n';
//   buffer[len * 12 + 1] = '\0';
// }

// }  // namespace cute

#pragma once

#include <cute/tensor.hpp>
#include <cstdio>

namespace onnxruntime {
namespace flash {

using namespace cute;

// Dispatch for 1D tensor
template <typename T>
__device__ void print_tensor_dispatch(const T& tensor, std::integral_constant<int, 1>) {
  printf("Tensor (1D), size: %d\n", (int)tensor.size());
  for (int i = 0; i < tensor.size(); ++i) {
    printf("%f ", (float)tensor(i));
  }
  printf("\n");
}

// Dispatch for 2D tensor
template <typename T>
__device__ void print_tensor_dispatch(const T& tensor, std::integral_constant<int, 2>) {
  printf("Tensor (2D), size: %d x %d\n", (int)size<0>(tensor), (int)size<1>(tensor));
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      printf("%f ", (float)tensor(mi, ni));
    }
    printf("\n");
  }
}

// Fallback for rank > 2
template <typename T, int Rank>
__device__ void print_tensor_dispatch(const T& tensor, std::integral_constant<int, Rank>) {
  printf("Tensor printing not supported for rank > 2 (rank = %d).\n", Rank);
}

// Unified API
template <typename T>
__device__ void print_tensor(const T& tensor) {
  constexpr int rank = decltype(tensor.shape())::rank;
  print_tensor_dispatch(tensor, std::integral_constant<int, rank>{});
}

}  // namespace flash
}  // namespace onnxruntime
