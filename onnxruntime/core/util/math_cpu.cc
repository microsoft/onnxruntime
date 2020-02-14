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

#include <algorithm>
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#else
#pragma warning(push)
#pragma warning(disable : 4267)
#pragma warning(disable : 4127)
#endif
#include "Eigen/src/Core/arch/Default/Half.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif
using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace math {

// MatMul implementation purely based on Eigen.
#define EIGEN_MATMUL_FUNCTION(T)                                                                \
  template <>                                                                                   \
  void MatMul<T>(int M, int N, int K, const T* A, const T* B, T* C, concurrency::ThreadPool*) { \
    auto C_mat = EigenMatrixMap<T>(C, N, M);                                                    \
    C_mat.noalias() = ConstEigenMatrixMap<T>(B, N, K) * ConstEigenMatrixMap<T>(A, K, M);        \
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
// when USE_MKLML is defined, use cblas APIs for MKLML
#if !defined(USE_MKLML_FOR_BLAS)

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
void Gemm<float, ThreadPool>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int64_t M,
                             const int64_t N, const int64_t K, float alpha, const float* A, const float* B, float beta,
                             float* C, ThreadPool* threadpool) {
  int lda = static_cast<int>((TransA == CblasNoTrans) ? K : M);
  int ldb = static_cast<int>((TransB == CblasNoTrans) ? N : K);
  MlasGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N, threadpool);
}

template <>
void MatMul<float>(int M, int N, int K, const float* A, const float* B, float* C, ThreadPool* threadpool) {
  MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.f, A, K, B, N, 0.f, C, N, threadpool);
}

#if defined(_M_AMD64) || defined(__x86_64__)
template <>
void MatMul<double>(int M, int N, int K, const double* A, const double* B, double* C, ThreadPool* threadpool) {
  MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.f, A, K, B, N, 0.f, C, N, threadpool);
}
#else
EIGEN_MATMUL_FUNCTION(double)
#endif

template <>
void GemmEx<float, ThreadPool>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, int M, int N, int K,
                               float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
                               int ldc, ThreadPool* threadpool) {
  MlasGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, threadpool);
}

template <>
void Gemv<float, CPUMathUtil>(const CBLAS_TRANSPOSE TransA, int M, int N, float alpha, const float* A, const float* x,
                              float beta, float* y, CPUMathUtil* /*provider*/) {
  EigenVectorMap<float> y_vec(y, TransA == CblasNoTrans ? M : N);
  if (beta == 0) {
    // In Caffe2 we often do a lazy initialization, which may contain NaNs in
    // the float values. As a result, if beta is 0, we explicitly do a setzero.
    y_vec.setZero();
  } else {
    y_vec *= beta;
  }
  switch (TransA) {
    case CblasNoTrans: {
      y_vec.noalias() += alpha * (ConstEigenMatrixMap<float>(A, N, M).transpose() *
                                  ConstEigenVectorMap<float>(x, N));
      return;
    }
    case CblasTrans: {
      y_vec.noalias() += alpha * (ConstEigenMatrixMap<float>(A, N, M) *
                                  ConstEigenVectorMap<float>(x, M));
      return;
    }
    default:
      ORT_THROW("Gemv float found an unexpected CBLAS_TRANSPOSE input of", TransA);
  }
}

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

#else

template <>
void Gemm<float, ThreadPool>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int64_t M,
                             const int64_t N, const int64_t K, float alpha, const float* A, const float* B, float beta,
                             float* C, ThreadPool* /*context*/) {
  int lda = gsl::narrow_cast<int>((TransA == CblasNoTrans) ? K : M);
  int ldb = gsl::narrow_cast<int>((TransB == CblasNoTrans) ? N : K);
  cblas_sgemm(CblasRowMajor, TransA, TransB,
              gsl::narrow_cast<int>(M),
              gsl::narrow_cast<int>(N),
              gsl::narrow_cast<int>(K),
              alpha, A, lda, B, ldb,
              beta, C, gsl::narrow_cast<int>(N));
}

template <>
void MatMul<float>(int M, int N, int K, const float* A, const float* B, float* C, ThreadPool*) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
}

template <>
void MatMul<double>(int M, int N, int K, const double* A, const double* B, double* C, ThreadPool*) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
}

template <>
void GemmEx<float, ThreadPool>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, int M, int N, int K,
                               float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
                               int ldc, ThreadPool* /*context*/) {
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template <>
void Gemv<float, CPUMathUtil>(const CBLAS_TRANSPOSE TransA, int M, int N, float alpha, const float* A, const float* x,
                              float beta, float* y, CPUMathUtil* /*context*/) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

#define CAFFE2_SPECIALIZED_AXPY(T, prefix)                                           \
  template <>                                                                        \
  void Axpy<T, CPUMathUtil>(int N, const T alpha, const T* x, T* y, CPUMathUtil*) {  \
    cblas_##prefix##axpy(N, alpha, x, 1, y, 1);                                      \
  }                                                                                  \
  template <>                                                                        \
  void Axpy<T, CPUMathUtil>(int N, const T* alpha, const T* x, T* y, CPUMathUtil*) { \
    cblas_##prefix##axpy(N, *alpha, x, 1, y, 1);                                     \
  }
CAFFE2_SPECIALIZED_AXPY(float, s)
#undef CAFFE2_SPECIALIZED_AXPY

#endif

#define DELEGATE_SIMPLE_UNARY_FUNCTION(T, Funcname, expr)                  \
  template <>                                                              \
  void Funcname<T, CPUMathUtil>(int N, const T* x, T* y, CPUMathUtil*) {   \
    EigenVectorMap<T>(y, N) = ConstEigenVectorMap<T>(x, N).array().expr(); \
  }
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Exp, exp)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqr, square)
#undef DELEGATE_SIMPLE_UNARY_FUNCTION

#define DELEGATE_POWX_FUNCTION(T)                                          \
  template <>                                                              \
  void Powx<T, CPUMathUtil>(int N, const T* a, T b, T* y, CPUMathUtil*) {  \
    EigenVectorMap<T>(y, N) = ConstEigenVectorMap<T>(a, N).array().pow(b); \
  }
DELEGATE_POWX_FUNCTION(float)
#undef DELEGATE_POWX_FUNCTION

#define EIGEN_SIMPLE_BINARY_FUNCTION(T, Funcname, expr)                                                       \
  template <>                                                                                                 \
  void Funcname<T, CPUMathUtil>(int N, const T* a, const T* b, T* y, CPUMathUtil*) {                          \
    EigenVectorMap<T>(y, N) = ConstEigenVectorMap<T>(a, N).array() expr ConstEigenVectorMap<T>(b, N).array(); \
  }

#define DEFINE_SIMPLE_BINARY_FUNCTION(Funcname, expr)   \
  EIGEN_SIMPLE_BINARY_FUNCTION(float, Funcname, expr)   \
  EIGEN_SIMPLE_BINARY_FUNCTION(int32_t, Funcname, expr) \
  EIGEN_SIMPLE_BINARY_FUNCTION(int64_t, Funcname, expr)

DEFINE_SIMPLE_BINARY_FUNCTION(Add, +)
DEFINE_SIMPLE_BINARY_FUNCTION(Mul, *)

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
#undef SPECIALIZED_ROWWISEMAX

#define SPECIALIZED_SET(T)                                                       \
  template <>                                                                    \
  void Set<T, CPUMathUtil>(const int64_t N, const T alpha, T* Y, CPUMathUtil*) { \
    if (alpha == (T)0) {                                                         \
      memset(Y, 0, N * sizeof(T));                                               \
    } else {                                                                     \
      EigenVectorMap<T>(Y, N).setConstant(alpha);                                \
    }                                                                            \
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

template <typename T>
static void Im2colWithEqualPadding(int64_t output_h, int64_t output_w, const T* data_im, int64_t channels,
                                   int64_t height, int64_t width, int64_t kernel_h, int64_t kernel_w,
                                   int64_t dilation_h, int64_t dilation_w, int64_t pad_t, int64_t pad_l,
                                   int64_t stride_h, int64_t stride_w, T* data_col, T padding_value) {
  // From Intel, https://github.com/BVLC/caffe/pull/3536
  int64_t pad_h = pad_t;
  int64_t pad_w = pad_l;
  int64_t channel_size = height * width;
  for (int64_t channel = channels; channel--; data_im += channel_size) {
    for (int64_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int64_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int64_t input_row = -pad_h + kernel_row * dilation_h;
        for (int64_t output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            std::fill_n(data_col, output_w, padding_value);
            data_col += output_w;
          } else {
            int64_t input_col = -pad_w + kernel_col * dilation_w;
            const T* rdptr = data_im + input_row * width + input_col;
            for (int64_t i = 0; i != output_w; ++i) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = rdptr[i * stride_w];
              } else {
                *(data_col++) = padding_value;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <typename T>
void Im2col<T, StorageOrder::NCHW>::operator()(const T* data_im, int64_t channels, int64_t height,
                                               int64_t width, int64_t kernel_h, int64_t kernel_w,
                                               int64_t dilation_h, int64_t dilation_w, int64_t pad_t,
                                               int64_t pad_l, int64_t pad_b, int64_t pad_r, int64_t stride_h,
                                               int64_t stride_w, T* data_col, T padding_value) {
  const int64_t output_h =
      (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int64_t output_w =
      (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
      1;

  // Fast path for zero padding and no dilation
  // From Torch, THNN_(unfolded_copy)
  if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 &&
      pad_t == 0 && pad_b == 0) {
    for (auto k = 0; k < channels * kernel_h * kernel_w; k++) {
      const auto nip = k / (kernel_h * kernel_w);
      const auto rest = k % (kernel_h * kernel_w);
      const auto kh = rest / kernel_w;
      const auto kw = rest % kernel_w;
      auto* dst = data_col + nip * (kernel_h * kernel_w * output_h * output_w) +
                  kh * (kernel_w * output_h * output_w) + kw * (output_h * output_w);
      const auto* src = data_im + nip * (height * width);
      for (auto y = 0; y < output_h; y++) {
        const auto iy = y * stride_h + kh;
        const auto ix = kw;
        if (stride_w == 1) {
          memcpy(
              dst + (y * output_w),
              src + (iy * width + ix),
              sizeof(T) * output_w);
        } else {
          for (auto x = 0; x < output_w; x++) {
            memcpy(
                dst + (y * output_w + x),
                src + (iy * width + ix + x * stride_w),
                sizeof(T));
          }
        }
      }
    }
    return;
  }

  // Fast path for equal padding
  if (pad_l == pad_r && pad_t == pad_b) {
    Im2colWithEqualPadding(output_h, output_w, data_im, channels, height, width, kernel_h, kernel_w, dilation_h, dilation_w, pad_t, pad_l, stride_h, stride_w, data_col, padding_value);
    return;
  }

  // Baseline
  const int64_t dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int64_t dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int64_t height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int64_t width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  int64_t channels_col = channels * kernel_h * kernel_w;
  for (int64_t c = 0; c < channels_col; ++c) {
    int64_t w_offset = c % kernel_w;
    int64_t h_offset = (c / kernel_w) % kernel_h;
    int64_t c_im = c / kernel_h / kernel_w;
    for (int64_t h = 0; h < height_col; ++h) {
      for (int64_t w = 0; w < width_col; ++w) {
        int64_t h_pad = h * stride_h - pad_t + h_offset * dilation_h;
        int64_t w_pad = w * stride_w - pad_l + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
              data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = padding_value;
      }
    }
  }
}

template struct Im2col<float, StorageOrder::NCHW>;
template struct Im2col<uint8_t, StorageOrder::NCHW>;

template <>
void Im2col<float, StorageOrder::NHWC>::operator()(const float* data_im, int64_t channels, int64_t height,
                                                   int64_t width, int64_t kernel_h, int64_t kernel_w,
                                                   int64_t dilation_h, int64_t dilation_w, int64_t pad_t,
                                                   int64_t pad_l, int64_t pad_b, int64_t pad_r, int64_t stride_h,
                                                   int64_t stride_w, float* data_col, float padding_value) {
  const int64_t dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int64_t dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int64_t height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int64_t width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  int64_t h_pad = -pad_t;
  for (int64_t h = 0; h < height_col; ++h) {
    int64_t w_pad = -pad_l;
    for (int64_t w = 0; w < width_col; ++w) {
      for (int64_t ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h) {
        for (int64_t iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            memcpy(data_col, data_im + (ih * width + iw) * channels,
                   sizeof(float) * channels);
          } else {
            std::fill_n(data_col, channels, padding_value);
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
void Col2im<float, CPUMathUtil, StorageOrder::NCHW>(const float* data_col, int64_t channels, int64_t height,
                                                    int64_t width, int64_t kernel_h, int64_t kernel_w,
                                                    int64_t dilation_h, int64_t dilation_w, int64_t pad_t,
                                                    int64_t pad_l, int64_t pad_b, int64_t pad_r, int64_t stride_h,
                                                    int64_t stride_w, float* data_im, CPUMathUtil* context) {
  const int64_t output_h =
      (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int64_t output_w =
      (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
      1;

  Set<float, CPUMathUtil>(height * width * channels, 0, data_im, context);

  // Fast path for zero padding and no dilation
  // From Torch, modified THNN_(unfolded_acc)
  if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 &&
      pad_t == 0 && pad_b == 0) {
    for (auto k = 0; k < channels * kernel_h * kernel_w; k++) {
      const auto nip = k / (kernel_h * kernel_w);
      const auto rest = k % (kernel_h * kernel_w);
      const auto kh = rest / kernel_w;
      const auto kw = rest % kernel_w;
      const auto* dst = data_col +
                        nip * (kernel_h * kernel_w * output_h * output_w) +
                        kh * (kernel_w * output_h * output_w) + kw * (output_h * output_w);
      auto* src = data_im + nip * (height * width);
      for (auto y = 0; y < output_h; y++) {
        const auto iy = y * stride_h + kh;
        const auto ix = kw;
        if (stride_w == 1) {
          auto offsrc = src + (iy * width + ix);
          const auto offdst = dst + (y * output_w);
          for (auto i = 0; i < output_w; ++i) {
            offsrc[i] += offdst[i];
          }
        } else {
          for (auto x = 0; x < output_w; x++) {
            auto offsrc = src + (iy * width + ix + x * stride_w);
            const auto offdst = dst + (y * output_w + x);
            *offsrc += *offdst;
          }
        }
      }
    }
    return;
  }

  // Fast path for equal padding
  if (pad_l == pad_r && pad_t == pad_b) {
    // From Intel, https://github.com/BVLC/caffe/pull/3536
    const int64_t pad_h = pad_t;
    const int64_t pad_w = pad_l;
    const int64_t channel_size = height * width;
    for (int64_t channel = channels; channel--; data_im += channel_size) {
      for (int64_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int64_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int64_t input_row = -pad_h + kernel_row * dilation_h;
          for (int64_t output_rows = output_h; output_rows; output_rows--) {
            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
              data_col += output_w;
            } else {
              int64_t input_col = -pad_w + kernel_col * dilation_w;
              for (int64_t output_col = output_w; output_col; output_col--) {
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  data_im[input_row * width + input_col] += *data_col;
                }
                data_col++;
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
    return;
  }

  // Fallback
  const int64_t dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int64_t dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int64_t height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int64_t width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  int64_t channels_col = channels * kernel_h * kernel_w;
  for (int64_t c = 0; c < channels_col; ++c) {
    int64_t w_offset = c % kernel_w;
    int64_t h_offset = (c / kernel_w) % kernel_h;
    int64_t c_im = c / kernel_h / kernel_w;
    for (int64_t h = 0; h < height_col; ++h) {
      for (int64_t w = 0; w < width_col; ++w) {
        int64_t h_pad = h * stride_h - pad_t + h_offset * dilation_h;
        int64_t w_pad = w * stride_w - pad_l + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
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

  Set<float, CPUMathUtil>(height * width * channels, 0, data_im, context);
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
                                                      const int64_t* col_shape, int64_t img_size, int64_t col_size,
                                                      const int64_t* kernel_shape, const int64_t* stride,
                                                      const int64_t* dilation, const int64_t* pad, int64_t N,
                                                      float* data_img, CPUMathUtil* context) {
  Set<float, CPUMathUtil>(img_size, 0, data_img, context);
  Im2colNd<float, StorageOrder::NCHW>()(
      data_col,
      img_shape,
      col_shape,
      img_size,
      col_size,
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

uint16_t floatToHalf(float f) {
  return Eigen::half_impl::float_to_half_rtne(f).x;
}

float halfToFloat(uint16_t h) {
  return Eigen::half_impl::half_to_float(Eigen::half_impl::raw_uint16_to_half(h));
}

}  // namespace math
}  // namespace onnxruntime
