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
// Modifications Copyright (c) Microsoft.

#include "core/util/math_cpuonly.h"
#include "core/util/math.h"
#include "core/framework/float16.h"

#include <algorithm>
#include <type_traits>
#include "core/common/narrow.h"
#include "core/mlas/inc/mlas.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "Eigen/src/Core/arch/Default/Half.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace math {

// MatMul implementation purely based on Eigen.
#define EIGEN_MATMUL_FUNCTION(T)                                                                                  \
  template <>                                                                                                     \
  void MatMul<T>(ptrdiff_t M, ptrdiff_t N, ptrdiff_t K, const T* A, const T* B, T* C, concurrency::ThreadPool*) { \
    auto C_mat = EigenMatrixMap<T>(C, N, M);                                                                      \
    C_mat.noalias() = ConstEigenMatrixMap<T>(B, N, K) * ConstEigenMatrixMap<T>(A, K, M);                          \
  }

EIGEN_MATMUL_FUNCTION(int32_t)
EIGEN_MATMUL_FUNCTION(uint32_t)
EIGEN_MATMUL_FUNCTION(int64_t)
EIGEN_MATMUL_FUNCTION(uint64_t)

////////////////////////////////////////////////////////////////////////////////
// BLAS alternatives.
// Depending on whether we have specified an external BLAS library or not, we
// will delegate the Caffe math functions that are BLAS-related to either the
// CBLAS call or the Eigen implementation.
////////////////////////////////////////////////////////////////////////////////

// Caffe2 gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
//
// The gemm call implements the following operation:
//
//                  C = alpha * op(A) * op(B) + beta * C
//
// where op(A) has size M x K, op(B) has size K x N, and C has size M x N. Each
// of A, B, and C are matrices and alpha and beta are scalars. Note that the
// most common use case of gemm will involve setting alpha to 1 and beta to 0.
//
// op(A) and op(B) represent the transformations that are done to A and B before
// the matrix multiply; depending on the flags set, op(A) is equal to A or A^T
// (transpose) if the argument TransA or TransB is set to CblasNoTrans or
// CblasTrans, respectively, for each of A and B.
template <>
void Gemm<float, ThreadPool>(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, ptrdiff_t M,
                             ptrdiff_t N, ptrdiff_t K, float alpha, const float* A, const float* B, float beta,
                             float* C, ThreadPool* threadpool) {
  int lda = static_cast<int>((TransA == CblasNoTrans) ? K : M);
  int ldb = static_cast<int>((TransB == CblasNoTrans) ? N : K);
#if 0 // comment for prototype
  MlasGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N, threadpool);
#endif
}

template <>
void Gemm<Eigen::half, ThreadPool>(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, ptrdiff_t M,
                                   ptrdiff_t N, ptrdiff_t K, Eigen::half alpha, const Eigen::half* A, const Eigen::half* B, Eigen::half beta,
                                   Eigen::half* C, ThreadPool*) {
  auto C_mat = EigenMatrixMap<Eigen::half>(C, N, M);
  if (beta == static_cast<Eigen::half>(0)) {
    C_mat.setZero();
  } else {
    C_mat *= beta;
  }
  switch (TransA) {
    case CblasNoTrans: {
      switch (TransB) {
        case CblasNoTrans:
          C_mat.noalias() += alpha * (ConstEigenMatrixMap<Eigen::half>(B, N, K) *
                                      ConstEigenMatrixMap<Eigen::half>(A, K, M));
          return;
        case CblasTrans:
          C_mat.noalias() += alpha * (ConstEigenMatrixMap<Eigen::half>(B, K, N).transpose() *
                                      ConstEigenMatrixMap<Eigen::half>(A, K, M));
          return;
        default:
          ORT_THROW("CblasNoTrans Unexpected CBLAS_TRANSPOSE for TransB of ", TransB);
      }
    }
    case CblasTrans: {
      switch (TransB) {
        case CblasNoTrans:
          C_mat.noalias() += alpha * (ConstEigenMatrixMap<Eigen::half>(B, N, K) *
                                      ConstEigenMatrixMap<Eigen::half>(A, M, K).transpose());
          return;
        case CblasTrans:
          C_mat.noalias() += alpha * (ConstEigenMatrixMap<Eigen::half>(B, K, N).transpose() *
                                      ConstEigenMatrixMap<Eigen::half>(A, M, K).transpose());
          return;
        default:
          ORT_THROW("CblasTrans Unexpected CBLAS_TRANSPOSE for TransB of ", TransB);
      }
    }
    default:
      ORT_THROW("Unexpected CBLAS_TRANSPOSE for TransA of ", TransA);
  }
}

#ifdef MLAS_SUPPORTS_GEMM_DOUBLE
template <>
void Gemm<double, ThreadPool>(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, ptrdiff_t M,
                              ptrdiff_t N, ptrdiff_t K, double alpha, const double* A, const double* B, double beta,
                              double* C, ThreadPool* threadpool) {
  int lda = static_cast<int>((TransA == CblasNoTrans) ? K : M);
  int ldb = static_cast<int>((TransB == CblasNoTrans) ? N : K);
  MlasGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N, threadpool);
}
#else
template <>
void Gemm<double, ThreadPool>(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, ptrdiff_t M,
                              ptrdiff_t N, ptrdiff_t K, double alpha, const double* A, const double* B, double beta,
                              double* C, ThreadPool*) {
  auto C_mat = EigenMatrixMap<double>(C, N, M);
  if (beta == 0) {
    C_mat.setZero();
  } else {
    C_mat *= beta;
  }
  switch (TransA) {
    case CblasNoTrans: {
      switch (TransB) {
        case CblasNoTrans:
          C_mat.noalias() += alpha * (ConstEigenMatrixMap<double>(B, N, K) *
                                      ConstEigenMatrixMap<double>(A, K, M));
          return;
        case CblasTrans:
          C_mat.noalias() += alpha * (ConstEigenMatrixMap<double>(B, K, N).transpose() *
                                      ConstEigenMatrixMap<double>(A, K, M));
          return;
        default:
          ORT_THROW("CblasNoTrans Unexpected CBLAS_TRANSPOSE for TransB of ", TransB);
      }
    }
    case CblasTrans: {
      switch (TransB) {
        case CblasNoTrans:
          C_mat.noalias() += alpha * (ConstEigenMatrixMap<double>(B, N, K) *
                                      ConstEigenMatrixMap<double>(A, M, K).transpose());
          return;
        case CblasTrans:
          C_mat.noalias() += alpha * (ConstEigenMatrixMap<double>(B, K, N).transpose() *
                                      ConstEigenMatrixMap<double>(A, M, K).transpose());
          return;
        default:
          ORT_THROW("CblasTrans Unexpected CBLAS_TRANSPOSE for TransB of ", TransB);
      }
    }
    default:
      ORT_THROW("Unexpected CBLAS_TRANSPOSE for TransA of ", TransA);
  }
}
#endif

template <>
void MatMul<float>(ptrdiff_t M, ptrdiff_t N, ptrdiff_t K, const float* A, const float* B, float* C, ThreadPool* threadpool) {
#if 0 // comment for prototype
  MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.f, A, K, B, N, 0.f, C, N, threadpool);
#endif
}

#ifdef MLAS_SUPPORTS_GEMM_DOUBLE
template <>
void MatMul<double>(ptrdiff_t M, ptrdiff_t N, ptrdiff_t K, const double* A, const double* B, double* C, ThreadPool* threadpool) {
  MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.f, A, K, B, N, 0.f, C, N, threadpool);
}
#else
EIGEN_MATMUL_FUNCTION(double)
#endif

template <>
void GemmEx<float, ThreadPool>(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, ptrdiff_t M, ptrdiff_t N, ptrdiff_t K,
                               float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
                               int ldc, ThreadPool* threadpool) {
#if 0 // comment for prototype
  MlasGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, threadpool);
#endif
}

template <typename T, class Provider>
void Gemv(CBLAS_TRANSPOSE TransA,
          int M,
          int N,
          float alpha,
          const T* A,
          const T* x,
          float beta,
          T* y,
          Provider* /*provider*/) {
  EigenVectorMap<T> y_vec(y, TransA == CblasNoTrans ? M : N);
  if (beta == 0) {
    // In Caffe2 we often do a lazy initialization, which may contain NaNs in
    // the float-pointing values. As a result, if beta is 0, we explicitly do a setzero.
    y_vec.setZero();
  } else {
    y_vec *= beta;
  }
  switch (TransA) {
    case CblasNoTrans: {
      y_vec.noalias() += alpha * (ConstEigenMatrixMap<T>(A, N, M).transpose() *
                                  ConstEigenVectorMap<T>(x, N));
      return;
    }
    case CblasTrans: {
      y_vec.noalias() += alpha * (ConstEigenMatrixMap<T>(A, N, M) *
                                  ConstEigenVectorMap<T>(x, M));
      return;
    }
    default:
      ORT_THROW("Gemv found an unexpected CBLAS_TRANSPOSE input of", TransA);
  }
}

template void Gemv<float, CPUMathUtil>(const CBLAS_TRANSPOSE TransA, int M, int N, float alpha, const float* A, const float* x,
                                       float beta, float* y, CPUMathUtil*);
template void Gemv<double, CPUMathUtil>(const CBLAS_TRANSPOSE TransA, int M, int N, float alpha, const double* A, const double* x,
                                        float beta, double* y, CPUMathUtil*);
#define SPECIALIZED_AXPY(T)                                                                       \
  template <>                                                                                     \
  void Axpy<T, CPUMathUtil>(int N, const T alpha, const T* x, T* Y, CPUMathUtil* /*provider*/) {  \
    EigenVectorMap<T>(Y, N) += ConstEigenVectorMap<T>(x, N) * alpha;                              \
  }                                                                                               \
  template <>                                                                                     \
  void Axpy<T, CPUMathUtil>(int N, const T* alpha, const T* x, T* Y, CPUMathUtil* /*provider*/) { \
    EigenVectorMap<T>(Y, N) += ConstEigenVectorMap<T>(x, N) * (*alpha);                           \
  }
SPECIALIZED_AXPY(float)
#undef SPECIALIZED_AXPY

#define DELEGATE_SIMPLE_UNARY_FUNCTION(T, Funcname, expr)                      \
  template <>                                                                  \
  void Funcname<T, CPUMathUtil>(ptrdiff_t N, const T* x, T* y, CPUMathUtil*) { \
    EigenVectorMap<T>(y, N) = ConstEigenVectorMap<T>(x, N).array().expr();     \
  }
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Exp, exp)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Exp, exp)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Log, log)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqr, square)
#undef DELEGATE_SIMPLE_UNARY_FUNCTION

#define EIGEN_SIMPLE_BINARY_FUNCTION(T, Funcname, expr)                                                       \
  template <>                                                                                                 \
  void Funcname<T, CPUMathUtil>(ptrdiff_t N, const T* a, const T* b, T* y, CPUMathUtil*) {                    \
    EigenVectorMap<T>(y, N) = ConstEigenVectorMap<T>(a, N).array() expr ConstEigenVectorMap<T>(b, N).array(); \
  }

#define DEFINE_SIMPLE_BINARY_FUNCTION(Funcname, expr)   \
  EIGEN_SIMPLE_BINARY_FUNCTION(float, Funcname, expr)   \
  EIGEN_SIMPLE_BINARY_FUNCTION(int32_t, Funcname, expr) \
  EIGEN_SIMPLE_BINARY_FUNCTION(int64_t, Funcname, expr)

DEFINE_SIMPLE_BINARY_FUNCTION(Add, +)
DEFINE_SIMPLE_BINARY_FUNCTION(Sub, -)
DEFINE_SIMPLE_BINARY_FUNCTION(Mul, *)
DEFINE_SIMPLE_BINARY_FUNCTION(Div, /)

#undef EIGEN_SIMPLE_BINARY_FUNCTION
#undef DEFINE_FLOAT_BINARY_FUNCTION

////////////////////////////////////////////////////////////////////////////////
// common math functions being used in Caffe that do not have a BLAS or MKL
// equivalent. For all these functions, we will simply implement them either via
// Eigen or via custom code.
////////////////////////////////////////////////////////////////////////////////

#define SPECIALIZED_ROWWISEMAX(T)                                                   \
  template <>                                                                       \
  void RowwiseMax<T, CPUMathUtil>(int N, int D, const T* x, T* y, CPUMathUtil*) {   \
    EigenVectorMap<T>(y, N) = ConstEigenMatrixMap<T>(x, D, N).colwise().maxCoeff(); \
  }
SPECIALIZED_ROWWISEMAX(float)
SPECIALIZED_ROWWISEMAX(double)
#undef SPECIALIZED_ROWWISEMAX

#define SPECIALIZED_SET(T)                                                         \
  template <>                                                                      \
  void Set<T, CPUMathUtil>(const ptrdiff_t N, const T alpha, T* Y, CPUMathUtil*) { \
    if (alpha == (T)0) {                                                           \
      memset(Y, 0, N * sizeof(T));                                                 \
    } else {                                                                       \
      EigenVectorMap<T>(Y, N).setConstant(alpha);                                  \
    }                                                                              \
  }

SPECIALIZED_SET(float);
SPECIALIZED_SET(double);
SPECIALIZED_SET(int8_t);
SPECIALIZED_SET(int16_t);
SPECIALIZED_SET(int32_t);
SPECIALIZED_SET(int64_t);
SPECIALIZED_SET(bool);
SPECIALIZED_SET(char);
SPECIALIZED_SET(uint8_t);
SPECIALIZED_SET(uint16_t);
#undef SPECIALIZED_SET

// Loop over spatial axes in reverse order to choose an index, like counting.
static inline bool NextPosition(int64_t N, const int64_t* shape, int64_t* dims) {
  bool has_next_output = false;
  for (int64_t d_i = N - 1; d_i >= 0; --d_i) {
    int64_t d_max = shape[d_i];
    ORT_ENFORCE(dims[d_i] < d_max);
    if (dims[d_i] == d_max - 1) {
      dims[d_i] = 0;
    } else {  // dims[d_i] < d_max - 1
      ++dims[d_i];
      has_next_output = true;
      break;
    }
  }
  return has_next_output;
}

template <typename T>
void Im2col<T, StorageOrder::NCHW>::operator()(
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
    T padding_value) {
  const int64_t output_h = (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int64_t output_w = (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  // From Intel, https://github.com/BVLC/caffe/pull/3536
  int64_t channel_size = height * width;
  for (int64_t channel = channels; channel--; data_im += channel_size) {
    for (int64_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int64_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int64_t input_row = -pad_t + kernel_row * dilation_h;
        for (int64_t output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            std::fill_n(data_col, output_w, padding_value);
            data_col += output_w;
          } else {
            int64_t input_col = -pad_l + kernel_col * dilation_w;
            const T* rdptr = data_im + input_row * width + input_col;
            for (int64_t i = 0; i < output_w;) {
              int64_t output_handled = 1;
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                if (stride_w == 1) {
                  // Compute the minimum of the number of input elements remaining
                  // and the number of output elements to produce.
                  output_handled = std::min(width - input_col, output_w - i);
                  data_col = std::copy_n(&rdptr[i], static_cast<size_t>(output_handled), data_col);
                } else if (stride_w == 2) {
                  // Same as above except using the number of strided input elements.
                  output_handled = std::min((width - input_col + 1) / 2, output_w - i);
                  const T* local_rdptr = &rdptr[i * 2];
                  for (int64_t x = output_handled; x > 0; x--) {
                    *(data_col++) = *local_rdptr;
                    local_rdptr += 2;
                  }
                } else {
                  *(data_col++) = rdptr[i * stride_w];
                }
              } else {
                *(data_col++) = padding_value;
              }
              input_col += output_handled * stride_w;
              i += output_handled;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <typename T>
void Im2col<T, StorageOrder::NCHW>::operator()(
    const T* data_im,
    const int64_t* im_shape,
    const int64_t* output_shape,
    int64_t channels_col,
    const int64_t* kernel_shape,
    const int64_t* stride,
    const int64_t* dilation,
    const int64_t* pad,
    ptrdiff_t rank,
    T* data_col,
    bool accumulate_output,
    T padding_value) {
  int64_t kernel_size = std::accumulate(kernel_shape, kernel_shape + rank, 1LL, std::multiplies<int64_t>());
  std::vector<int64_t> d_offset(rank, 0);
  std::vector<int64_t> d_iter(rank, 0);
  for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int64_t offset = c_col;
    for (ptrdiff_t d_i = rank - 1; d_i >= 0; --d_i) {
      if (d_i < rank - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    do {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      int64_t index_col = c_col;
      int64_t index_im = c_col / kernel_size;
      bool is_padding = false;
      for (ptrdiff_t d_i = 0; d_i < rank; ++d_i) {
        int64_t d = d_iter[d_i];
        int64_t d_im = d * stride[d_i] - pad[d_i] + d_offset[d_i] * dilation[d_i];
        is_padding |= !is_a_ge_zero_and_a_lt_b(d_im, im_shape[d_i]);
        index_col *= output_shape[d_i];
        index_col += d;
        index_im *= im_shape[d_i];
        index_im += d_im;
      }
      if (!accumulate_output) {
        if (is_padding) {
          data_col[index_col] = padding_value;
        } else {
          data_col[index_col] = data_im[index_im];
        }
      } else if (!is_padding) {  // col2im
        data_col[index_im] += data_im[index_col];
      }
    } while (NextPosition(rank, output_shape, d_iter.data()));
  }  // for (int c = 0; c < channels_col; ++c) {
}

template struct Im2col<float, StorageOrder::NCHW>;
template struct Im2col<uint8_t, StorageOrder::NCHW>;

template <typename T>
void Im2col<T, StorageOrder::NHWC>::operator()(
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
    T padding_value) {
  int64_t mh = output_start / output_w;
  int64_t mw = output_start % output_w;
  for (int64_t mz = output_start; mz < output_start + output_count; mz++) {
    int64_t oh = mh * stride_h;
    int64_t ow = mw * stride_w;

    for (int64_t kh = 0; kh < kernel_h; kh++) {
      int64_t ih = kh * dilation_h + oh - pad_t;

      if (is_a_ge_zero_and_a_lt_b(ih, input_h)) {
        int64_t iw = ow - pad_l;
        if (dilation_w == 1 && group_channels == input_channels) {
          int64_t kw = kernel_w;
          while (kw > 0) {
            if (is_a_ge_zero_and_a_lt_b(iw, input_w)) {
              // Increase the copy count size to reduce the number of copy calls.
              int64_t batch_w = std::min(kw, input_w - iw);
              std::memcpy(data_col, data_im + (ih * input_w + iw) * group_channels, narrow<size_t>(sizeof(T) * batch_w * group_channels));
              data_col += batch_w * group_channels;
              iw += batch_w;
              kw -= batch_w;
            } else {
              data_col = std::fill_n(data_col, group_channels, padding_value);
              iw++;
              kw--;
            }
          }
        } else {
          for (int64_t kw = 0; kw < kernel_w; kw++) {
            if (is_a_ge_zero_and_a_lt_b(iw, input_w)) {
              // N.B. Using std::memcpy helped here over std::copy_n when doing a
              // transform for an image with a small number of group channels.
              std::memcpy(data_col, data_im + (ih * input_w + iw) * input_channels, narrow<size_t>(sizeof(T) * group_channels));
              data_col += group_channels;
            } else {
              data_col = std::fill_n(data_col, group_channels, padding_value);
            }
            iw += dilation_w;
          }
        }
      } else {
        data_col = std::fill_n(data_col, kernel_w * group_channels, padding_value);
      }
    }

    if (++mw == output_w) {
      ++mh;
      mw = 0;
    }
  }
}

template <typename T>
void Im2col<T, StorageOrder::NHWC>::operator()(
    const T* data_im,
    int64_t group_channels,
    int64_t input_channels,
    const int64_t* im_shape,
    const int64_t* output_shape,
    const int64_t* kernel_shape,
    const int64_t* stride,
    const int64_t* dilation,
    const int64_t* pad,
    ptrdiff_t rank,
    T* data_col,
    T padding_value) {
  // iterate dimensions on output image shape (without Batch and Channel)
  std::vector<int64_t> d_output(rank, 0);
  // inner iterate dimensions on kernel shape (without output channel and input channel)
  std::vector<int64_t> d_kernel(rank, 0);

  // Loop over spatial axes along the output image shape
  do {
    // Loop over spatial axes in reverse order to choose an index on kernel dimensions
    do {
      // Loop over spatial axes in forward order to compute the indices in the image
      // and the inner col, and whether the index lies in the padding.
      int64_t index_im = 0;
      bool is_padding = false;
      for (ptrdiff_t d_i = 0; d_i < rank; ++d_i) {
        int64_t d_im = d_output[d_i] * stride[d_i] - pad[d_i] + d_kernel[d_i] * dilation[d_i];
        is_padding |= !is_a_ge_zero_and_a_lt_b(d_im, im_shape[d_i]);
        index_im *= im_shape[d_i];
        index_im += d_im;
      }
      index_im *= input_channels;

      if (is_padding) {
        data_col = std::fill_n(data_col, group_channels, padding_value);
      } else {
        data_col = std::copy_n(data_im + index_im, group_channels, data_col);
      }
    } while (NextPosition(rank, kernel_shape, d_kernel.data()));
  } while (NextPosition(rank, output_shape, d_output.data()));
}

template <typename T>
void Im2col<T, StorageOrder::NHWC>::operator()(
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
    const T* padding_ptr) {
  if (rank == 1) {
    int64_t stride_w = stride[0];
    int64_t kernel_w = kernel_shape[0];
    int64_t dilation_w = dilation[0];
    int64_t pad_l = pad[0];
    int64_t input_w = input_shape[0];

    int64_t ow = output_start * stride_w;

    while (output_count--) {
      int64_t iw = ow - pad_l;
      for (int64_t kw = 0; kw < kernel_w; kw++) {
        const T* data_ptr = data_im + iw * input_channels;
        data_indirection[kw] = is_a_ge_zero_and_a_lt_b(iw, input_w) ? data_ptr : padding_ptr;
        iw += dilation_w;
      }
      data_indirection += kernel_w;
      ow += stride_w;
    }

  } else if (rank == 2) {
    int64_t stride_h = stride[0];
    int64_t stride_w = stride[1];
    int64_t kernel_h = kernel_shape[0];
    int64_t kernel_w = kernel_shape[1];
    int64_t dilation_h = dilation[0];
    int64_t dilation_w = dilation[1];
    int64_t pad_t = pad[0];
    int64_t pad_l = pad[1];
    int64_t input_h = input_shape[0];
    int64_t input_w = input_shape[1];
    int64_t output_w = output_shape[1];

    int64_t oh = (output_start / output_w) * stride_h;
    int64_t ow = (output_start % output_w) * stride_w;
    int64_t ow_end = output_w * stride_w;

    while (output_count--) {
      for (int64_t kh = 0; kh < kernel_h; kh++) {
        int64_t ih = kh * dilation_h + oh - pad_t;
        if (is_a_ge_zero_and_a_lt_b(ih, input_h)) {
          int64_t ihw = ih * input_w;
          int64_t iw = ow - pad_l;
          if (kernel_w == 3) {
            const T* data_ptr0 = data_im + (ihw + iw) * input_channels;
            data_indirection[0] = is_a_ge_zero_and_a_lt_b(iw, input_w) ? data_ptr0 : padding_ptr;
            iw += dilation_w;
            const T* data_ptr1 = data_im + (ihw + iw) * input_channels;
            data_indirection[1] = is_a_ge_zero_and_a_lt_b(iw, input_w) ? data_ptr1 : padding_ptr;
            iw += dilation_w;
            const T* data_ptr2 = data_im + (ihw + iw) * input_channels;
            data_indirection[2] = is_a_ge_zero_and_a_lt_b(iw, input_w) ? data_ptr2 : padding_ptr;
          } else {
            for (int64_t kw = 0; kw < kernel_w; kw++) {
              const T* data_ptr = data_im + (ihw + iw) * input_channels;
              data_indirection[kw] = is_a_ge_zero_and_a_lt_b(iw, input_w) ? data_ptr : padding_ptr;
              iw += dilation_w;
            }
          }
        } else {
          for (int64_t kw = 0; kw < kernel_w; kw++) {
            data_indirection[kw] = padding_ptr;
          }
        }
        data_indirection += kernel_w;
      }
      ow += stride_w;
      if (ow == ow_end) {
        oh += stride_h;
        ow = 0;
      }
    }

  } else {
    // iterate dimensions on output image shape (without Batch and Channel)
    std::vector<int64_t> d_output(rank, 0);
    // inner iterate dimensions on kernel shape (without output channel and input channel)
    std::vector<int64_t> d_kernel(rank, 0);

    // Skip ahead to the starting output index.
    for (ptrdiff_t d_i = rank - 1; d_i >= 0; --d_i) {
      d_output[d_i] = output_start % output_shape[d_i];
      output_start /= output_shape[d_i];
    }

    while (output_count--) {
      // Loop over spatial axes in reverse order to choose an index on kernel dimensions
      do {
        // Loop over spatial axes in forward order to compute the indices in the image
        // and the inner col, and whether the index lies in the padding.
        int64_t index_im = 0;
        bool is_padding = false;
        for (ptrdiff_t d_i = 0; d_i < rank; ++d_i) {
          int64_t d_input = d_output[d_i] * stride[d_i] - pad[d_i] + d_kernel[d_i] * dilation[d_i];
          is_padding |= !is_a_ge_zero_and_a_lt_b(d_input, input_shape[d_i]);
          index_im *= input_shape[d_i];
          index_im += d_input;
        }
        const T* data_ptr = data_im + index_im * input_channels;
        *data_indirection++ = is_padding ? padding_ptr : data_ptr;
      } while (NextPosition(rank, kernel_shape, d_kernel.data()));
      // Loop over spatial axes along the output image shape
      NextPosition(rank, output_shape, d_output.data());
    }
  }
}

template struct Im2col<int8_t, StorageOrder::NHWC>;
template struct Im2col<uint8_t, StorageOrder::NHWC>;
template struct Im2col<MLFloat16, StorageOrder::NHWC>;

template <>
void Col2im<float, CPUMathUtil, StorageOrder::NCHW>(const float* data_col, int64_t channels, int64_t height,
                                                    int64_t width, int64_t kernel_h, int64_t kernel_w,
                                                    int64_t dilation_h, int64_t dilation_w, int64_t pad_t,
                                                    int64_t pad_l, int64_t pad_b, int64_t pad_r, int64_t stride_h,
                                                    int64_t stride_w, float* data_im, CPUMathUtil* context) {
  const int64_t output_h =
      (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int64_t output_w =
      (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int64_t output_hw = output_h * output_w;
  const int64_t hw = height * width;
  const int64_t hwc = hw * channels;
  Set<float, CPUMathUtil>(narrow<ptrdiff_t>(hwc), 0, data_im, context);

  // Fast path for zero padding and no dilation
  // From Torch, modified THNN_(unfolded_acc)
  if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 && pad_t == 0 && pad_b == 0) {
    // Src (column) data cursor
    auto* src = data_col;
    // End of dst (image) data
    auto* dst_end = data_im + hwc;
    // Dst cursor step at end of row
    auto dst_row_step = stride_h * width - stride_w * output_w;
    // Dst channel data
    for (auto* dst_cb = data_im; dst_cb < dst_end; dst_cb += hw) {
      // First dst row for current kernel row
      auto* dst_hb = dst_cb;
      for (auto kh = 0; kh < kernel_h; ++kh, dst_hb += width) {
        // First dst element for current kernel element
        auto* dst_wb = dst_hb;
        for (auto kw = 0; kw < kernel_w; ++kw, ++dst_wb) {
          // Dst cursor
          auto* dst = dst_wb;
          // End of source data for kernel element
          for (auto* src_he = src + output_hw; src < src_he; dst += dst_row_step) {
            // End of source row
            auto* src_we = src + output_w;
            if (stride_w == 1) {
              for (; src < src_we; ++src, ++dst) {
                *dst += *src;
              }
            } else {
              for (; src < src_we; ++src, dst += stride_w) {
                *dst += *src;
              }
            }
          }
        }
      }
    }
    return;
  }

  // Fallback

  // Src (col data) cursor
  auto* src = data_col;
  // End of dst (image) data
  auto* dst_end = data_im + hwc;
  // Begin of src channel data
  for (auto* dst = data_im; dst < dst_end; dst += hw) {
    // Current kernel element starting vertical offset in dst data
    int64_t h_offset = -pad_t * width;
    int64_t h_offset_end = h_offset + kernel_h * dilation_h * width;
    for (; h_offset < h_offset_end; h_offset += dilation_h * width) {
      // Current kernel element starting horizontal offset in dst data
      int64_t w_offset = -pad_l;
      int64_t w_offset_end = w_offset + kernel_w * dilation_w;
      for (; w_offset < w_offset_end; w_offset += dilation_w) {
        // End of src channel data
        auto* src_ce = src + output_hw;
        // Dst row offset
        for (int64_t h = h_offset; src < src_ce; h += stride_h * width) {
          // End of src row data
          auto* src_we = src + output_w;
          if (is_a_ge_zero_and_a_lt_b(h, hw)) {
            for (int64_t w = w_offset; src < src_we; src++, w += stride_w) {
              if (is_a_ge_zero_and_a_lt_b(w, width)) {
                dst[h + w] += *src;
              }
            }
          } else {
            src = src_we;
          }
        }
      }
    }
  }
}

template <>
void Col2im<float, CPUMathUtil, StorageOrder::NHWC>(const float* data_col, int64_t channels, int64_t height,
                                                    int64_t width, int64_t kernel_h, int64_t kernel_w,
                                                    int64_t dilation_h, int64_t dilation_w, int64_t pad_t,
                                                    int64_t pad_l, int64_t pad_b, int64_t pad_r, int64_t stride_h,
                                                    int64_t stride_w, float* data_im, CPUMathUtil* context) {
  const int64_t dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int64_t dkernel_w = dilation_w * (kernel_w - 1) + 1;

  const int64_t hwc = height * width * channels;
  Set<float, CPUMathUtil>(narrow<ptrdiff_t>(hwc), 0, data_im, context);
  int64_t height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int64_t width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  int64_t h_pad = -pad_t;
  for (int64_t h = 0; h < height_col; ++h) {
    int64_t w_pad = -pad_l;
    for (int64_t w = 0; w < width_col; ++w) {
      for (int64_t ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h) {
        for (int64_t iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            auto* data_im_patch = data_im + (ih * width + iw) * channels;
            Add<float, CPUMathUtil>(
                static_cast<int>(channels), data_im_patch, data_col, data_im_patch, context);
          }
          data_col += channels;
        }
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

template <>
void Col2imNd<float, CPUMathUtil, StorageOrder::NCHW>(const float* data_col, const int64_t* img_shape,
                                                      const int64_t* output_shape, int64_t channels_col, int64_t img_size,
                                                      const int64_t* kernel_shape, const int64_t* stride,
                                                      const int64_t* dilation, const int64_t* pad, ptrdiff_t N,
                                                      float* data_img, CPUMathUtil* context) {
  Set<float, CPUMathUtil>(narrow<ptrdiff_t>(img_size), 0, data_img, context);
  Im2col<float, StorageOrder::NCHW>()(
      data_col,
      img_shape,
      output_shape,
      channels_col,
      kernel_shape,
      stride,
      dilation,
      pad,
      N,
      data_img,
      true);
}

#define SPECIALIZED_COPYVECTOR(T)                                                          \
  template <>                                                                              \
  void CopyVector<T, CPUMathUtil>(int N, const T* src, T* dst, CPUMathUtil* /*context*/) { \
    if (src != dst && N > 0) {                                                             \
      memcpy(dst, src, sizeof(T) * N);                                                     \
    }                                                                                      \
  }
SPECIALIZED_COPYVECTOR(float)
#undef SPECIALIZED_COPYVECTOR

// like C++20's std::bit_cast
// adapted from the example implementation here: https://en.cppreference.com/w/cpp/numeric/bit_cast
// TODO replace this with std::bit_cast when we move to C++20
template <typename Dst, typename Src>
static std::enable_if_t<
    sizeof(Src) == sizeof(Dst) &&
        std::is_trivially_copyable_v<Src> &&
        std::is_trivially_copyable_v<Dst> &&
        std::is_trivially_constructible_v<Dst>,
    Dst>
BitCast(const Src& src) {
  Dst dst;
  std::memcpy(&dst, &src, sizeof(dst));
  return dst;
}

uint16_t floatToHalf(float f) {
  return BitCast<uint16_t>(Eigen::half_impl::float_to_half_rtne(f).x);
}

uint16_t doubleToHalf(double f) {
  return BitCast<uint16_t>(Eigen::half_impl::float_to_half_rtne(static_cast<float>(f)).x);
}

float halfToFloat(uint16_t h) {
  return Eigen::half_impl::half_to_float(Eigen::half_impl::raw_uint16_to_half(h));
}

// AddToRow and AddToCol adds the corresponding row/col vector b to the matrix a
// of shape M x N. The actual implementation uses eigen which is column major,
// so notice the row/column swap in the actual implementation.
#define DELEGATE_BROADCAST_BINARY_FUNCTION(T, Funcname, expr)                                                    \
  template <>                                                                                                    \
  void Funcname##ToRow<T, CPUMathUtil>(int M, int N, const T* a, const T* b, T* y, CPUMathUtil*) {               \
    EigenArrayMap<T>(y, N, M) = ConstEigenArrayMap<T>(a, N, M).colwise() expr ConstEigenVectorArrayMap<T>(b, N); \
  }                                                                                                              \
  /* inplace versions */                                                                                         \
  template <>                                                                                                    \
  void Funcname##ToRow<T, CPUMathUtil>(int M, int N, const T* x, T* y, CPUMathUtil*) {                           \
    EigenArrayMap<T>(y, N, M).colwise() expr## = ConstEigenVectorArrayMap<T>(x, N);                              \
  }                                                                                                              \
  template <>                                                                                                    \
  void Funcname##ToCol<T, CPUMathUtil>(int M, int N, const T* x, T* y, CPUMathUtil*) {                           \
    EigenArrayMap<T>(y, N, M).rowwise() expr## = ConstEigenVectorArrayMap<T>(x, M).transpose();                  \
  }

#define DEFINE_BROADCAST_BINARY_FUNCTION(name, op)      \
  DELEGATE_BROADCAST_BINARY_FUNCTION(int32_t, name, op) \
  DELEGATE_BROADCAST_BINARY_FUNCTION(int64_t, name, op) \
  DELEGATE_BROADCAST_BINARY_FUNCTION(float, name, op)

DEFINE_BROADCAST_BINARY_FUNCTION(Add, +)
DEFINE_BROADCAST_BINARY_FUNCTION(Sub, -)
DEFINE_BROADCAST_BINARY_FUNCTION(Mul, *)
DEFINE_BROADCAST_BINARY_FUNCTION(Div, /)

#define SPECIALIZED_ROWWISESUM(T)                                                 \
  template <>                                                                     \
  void RowwiseSum<T, CPUMathUtil>(int N, int D, const T* x, T* y, CPUMathUtil*) { \
    EigenVectorMap<T>(y, N) = ConstEigenMatrixMap<T>(x, D, N).colwise().sum();    \
  }
SPECIALIZED_ROWWISESUM(float)
SPECIALIZED_ROWWISESUM(int32_t)
SPECIALIZED_ROWWISESUM(int64_t)
SPECIALIZED_ROWWISESUM(double)
#undef SPECIALIZED_ROWWISESUM

#define SPECIALIZED_SUM(T)                                                             \
  template <>                                                                          \
  void Sum<T, CPUMathUtil>(ptrdiff_t N, const T* x, T* y, CPUMathUtil* /* unused */) { \
    *y = ConstEigenVectorMap<T>(x, N).sum();                                           \
  }

SPECIALIZED_SUM(float);
SPECIALIZED_SUM(int32_t);
SPECIALIZED_SUM(int64_t);

#undef SPECIALIZED_SUM

#define SPECIALIZED_SCALE(T)                                                                                 \
  template <>                                                                                                \
  void Scale<T, CPUMathUtil>(ptrdiff_t n, float alpha, const T* x, T* y, CPUMathUtil* /*provider*/) {        \
    EigenVectorMap<T>(y, n) = ConstEigenVectorMap<T>(x, n) * alpha;                                          \
  }                                                                                                          \
  template <>                                                                                                \
  void Scale<T, CPUMathUtil>(ptrdiff_t n, const float* alpha, const T* x, T* y, CPUMathUtil* /*provider*/) { \
    EigenVectorMap<T>(y, n) = ConstEigenVectorMap<T>(x, n) * (*alpha);                                       \
  }
SPECIALIZED_SCALE(float)
#undef SPECIALIZED_SCALE

#define SPECIALIZED_DOT(T)                                                                   \
  template <>                                                                                \
  void Dot<T, CPUMathUtil>(int N, const T* a, const T* b, T* y, CPUMathUtil* /*provider*/) { \
    *y = ConstEigenVectorMap<T>(a, N).dot(ConstEigenVectorMap<T>(b, N));                     \
  }
SPECIALIZED_DOT(float)
#undef SPECIALIZED_DOT

}  // namespace math
}  // namespace onnxruntime
