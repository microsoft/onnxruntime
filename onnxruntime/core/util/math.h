/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cassert>

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#endif

#ifndef CBLAS_ENUM_DEFINED_H
#define CBLAS_ENUM_DEFINED_H
enum CBLAS_ORDER { CblasRowMajor = 101,
                   CblasColMajor = 102 };
enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
};
enum CBLAS_UPLO { CblasUpper = 121,
                  CblasLower = 122 };
enum CBLAS_DIAG { CblasNonUnit = 131,
                  CblasUnit = 132 };
enum CBLAS_SIDE { CblasLeft = 141,
                  CblasRight = 142 };
#endif

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}

enum StorageOrder {
  UNKNOWN = 0,
  NHWC = 1,
  NCHW = 2,
};

namespace math {

template <typename T, class Provider>
void Exp(ptrdiff_t N, const T* x, T* y, Provider* provider);
template <typename T, class Provider>
void Log(ptrdiff_t N, const T* x, T* y, Provider* provider);
template <typename T, class Provider>
void Sqr(ptrdiff_t N, const T* x, T* y, Provider* provider);

#define DECLARE_BINARY_OP(name)                                                     \
  template <typename T, class Provider>                                             \
  void name(ptrdiff_t N, const T* a, const T* b, T* y, Provider* provider);         \
  template <typename T, class Provider>                                             \
  void name##ToRow(int M, int N, const T* a, const T* b, T* y, Provider* provider); \
  template <typename T, class Provider>                                             \
  void name##ToRow(int M, int N, const T* x, T* y, Provider* provider);             \
  template <typename T, class Provider>                                             \
  void name##ToCol(int M, int N, const T* x, T* y, Provider* provider);

DECLARE_BINARY_OP(Add);
DECLARE_BINARY_OP(Sub);
DECLARE_BINARY_OP(Mul);
DECLARE_BINARY_OP(Div);

#undef DECLARE_BINARY_OP

// Compute the row-wise max of a N*D matrix X, and write it to a N
// dimensional vector y.
template <typename T, class Provider>
void RowwiseMax(int N, int D, const T* x, T* y,
                Provider* provider);

// Compute the row-wise sum of a N*D matrix X, and write it to a N
// dimensional vector y.
template <typename T, class Provider>
void RowwiseSum(int N, int D, const T* x, T* y,
                Provider* provider);

// Sum of vector x, and writes the result to a single value y.
template <typename T, class Provider>
void Sum(ptrdiff_t N, const T* x, T* y, Provider* provider);

template <typename T, class Provider>
void Scale(ptrdiff_t N, float alpha, const T* x, T* y, Provider* provider);

// Different from the Scale function above, if alpha is passed in
// as a pointer, we will assume that it lives on the correct execution provider,
// for example on GPU.
template <typename T, class Provider>
void Scale(ptrdiff_t N, const float* alpha, const T* x, T* y, Provider* provider);

template <typename T>
void MatMul(
    ptrdiff_t M,
    ptrdiff_t N,
    ptrdiff_t K,
    const T* A,
    const T* B,
    T* C, concurrency::ThreadPool* threadpool);

// Decaf gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename T, class Provider>
void Gemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    ptrdiff_t M,
    ptrdiff_t N,
    ptrdiff_t K,
    T alpha,
    const T* A,
    const T* B,
    T beta,
    T* C,
    Provider*);

// We also provide a gemm that has explicit lda, ldb and ldc specified.
// In most cases you probably want to use the function above, though.
template <typename T, class Provider>
void GemmEx(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    ptrdiff_t M,
    ptrdiff_t N,
    ptrdiff_t K,
    T alpha,
    const T* A,
    int lda,
    const T* B,
    int ldb,
    T beta,
    T* C,
    int ldc,
    Provider*);

// Gemv always takes in a M*N matrix A, and depending on whether we set TransA
// to Trans, the output is:
// CblasNoTrans: x is an N dim vector and y is an M dim vector.
// CblasTrans:   x is an M dim vector and y is an N dim vector.
template <typename T, class Provider>
void Gemv(
    CBLAS_TRANSPOSE TransA,
    int M,
    int N,
    float alpha,
    const T* A,
    const T* x,
    float beta,
    T* y,
    Provider* provider);

template <typename T, class Provider>
void Set(ptrdiff_t N, T alpha, T* X, Provider* provider);

template <typename T, class Provider>
void Dot(int N, const T* a, const T* b, T* y, Provider* provider);

template <typename T, class Provider>
void Axpy(int N, float alpha, const T* x, T* y, Provider* provider);

// Different from the Axpy function above, if alpha is passed in
// as a pointer, we will assume that it lives on the correct execution provider,
// for example on GPU.
template <typename T, class Provider>
void Axpy(int N, const float* alpha, const T* x, T* y, Provider* provider);

template <typename T, int order>
struct Im2col {
};

template <typename T>
struct Im2col<T, StorageOrder::NCHW> {
  void operator()(
      const T* data_im,
      int64_t channels,
      int64_t height,
      int64_t width,
      int64_t kernel_h,
      int64_t kernel_w,
      int64_t dilation_h,
      int64_t dilation_w,
      int64_t pad_t,
      int64_t pad_l,
      int64_t pad_b,
      int64_t pad_r,
      int64_t stride_h,
      int64_t stride_w,
      T* data_col,
      T padding_value = 0);
  void operator()(
      const T* data_im,
      const int64_t* input_shape,
      const int64_t* output_shape,
      int64_t channels_col,
      const int64_t* kernel_shape,
      const int64_t* stride,
      const int64_t* dilation,
      const int64_t* pad,
      ptrdiff_t rank,
      T* data_col,
      bool accumulate_output = false,
      T padding_value = 0);
};

template <typename T>
struct Im2col<T, StorageOrder::NHWC> {
  void operator()(
      const T* data_im,
      int64_t group_channels,
      int64_t input_channels,
      int64_t input_h,
      int64_t input_w,
      int64_t kernel_h,
      int64_t kernel_w,
      int64_t dilation_h,
      int64_t dilation_w,
      int64_t pad_t,
      int64_t pad_l,
      int64_t stride_h,
      int64_t stride_w,
      int64_t output_w,
      int64_t output_start,
      int64_t output_count,
      T* data_col,
      T padding_value = 0);
  void operator()(
      const T* data_im,
      int64_t group_channels,
      int64_t input_channels,
      const int64_t* input_shape,
      const int64_t* output_shape,
      const int64_t* kernel_shape,
      const int64_t* stride,
      const int64_t* dilation,
      const int64_t* pad,
      ptrdiff_t rank,
      T* data_col,
      T padding_value = 0);
  void operator()(
      const T* data_im,
      int64_t input_channels,
      const int64_t* input_shape,
      const int64_t* output_shape,
      const int64_t* kernel_shape,
      const int64_t* stride,
      const int64_t* dilation,
      const int64_t* pad,
      ptrdiff_t rank,
      int64_t output_start,
      int64_t output_count,
      T const** data_indirection,
      const T* padding_ptr);
};

template <typename T, class Provider, int order>
void Col2imNd(
    const T* data_col,
    const int64_t* img_shape,
    const int64_t* output_shape,
    int64_t channels_col,
    int64_t img_size,
    const int64_t* kernel_shape,
    const int64_t* stride,
    const int64_t* dilation,
    const int64_t* pad,
    ptrdiff_t N,
    T* data_img,
    Provider* provider);

template <typename T, class Provider, int order>
void Col2im(
    const T* data_col,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t patch_h,
    int64_t patch_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t pad_t,
    int64_t pad_l,
    int64_t pad_b,
    int64_t pad_r,
    int64_t stride_h,
    int64_t stride_w,
    T* data_im,
    Provider* provider);

template <typename T, typename TypedCopy>
void CopyMatrix(
    int M,
    int N,
    const T* A,
    int lda,
    T* B,
    int ldb,
    TypedCopy copy) {
  {
    assert(M >= 0);
    assert(N >= 0);
    assert(lda >= 0);
    assert(ldb >= 0);
    if (lda == N && ldb == N) {
      copy(A, B, static_cast<size_t>(N) * static_cast<size_t>(M));
      return;
    }

    for (size_t i = 0; i < static_cast<size_t>(M); ++i) {
      copy(A + lda * i, B + ldb * i, static_cast<size_t>(N));
    }
  }
}

template <typename T, class Provider>
void CopyVector(int N, const T* A, T* B, Provider* provider);

// Function uses casting from int64_t to uint64_t to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always
// positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than
// 0x800...
// The casting allows to use one condition instead of two.
constexpr inline bool is_a_ge_zero_and_a_lt_b(int64_t a, int64_t b) {
  return static_cast<uint64_t>(a) < static_cast<uint64_t>(b);
}

// Calculates ceil(a / b). User must be careful to ensure that there
// is no overflow or underflow in the calculation.
template <typename T>
constexpr T divUp(T a, T b) {
  return (a + b - (T)1) / b;
}

// Rounds a up to the next highest multiple of b. User must be careful
// to ensure that there is no overflow or underflow in the calculation
// of divUp.
template <typename T>
constexpr T roundUp(T a, T b) {
  return divUp<T>(a, b) * b;
}

// Converts a float32 to a float16 value.
uint16_t floatToHalf(float f);

// Converts a double (float64) to a float16 value.
uint16_t doubleToHalf(double f);

// Converts a float16 to a float32 value.
float halfToFloat(uint16_t h);

}  // namespace math
}  // namespace onnxruntime
