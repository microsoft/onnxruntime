/**
* Derived from caffe2, need copy right announcement here.
*/

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

// disable noisy warning about use of std::copy_n by gsl::copy. gsl_algorithm does the same thing
// but that's too late for the disable to work
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include <algorithm>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "core/providers/cpu/math/softmax_shared.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

#include "gsl/gsl_algorithm"
#include "gsl/gsl_util"

namespace onnxruntime {

common::Status SoftmaxCPU(const int64_t N,
                          const int64_t D,
                          const float* Xdata,
                          float* Ydata,
                          float* scale,
                          const float* sum_multiplier,
                          bool logarithmic,
                          float* rowmax) {
  // the Math functions SoftmaxCPU uses only support int32_t as input, so enforce that
  if (N * D > INT32_MAX || N > INT32_MAX || D > INT32_MAX) {
    std::ostringstream ss;
    ss << "SoftmaxCPU inputs N, D and N * D must be < " << INT32_MAX << ". N=" << N << ", D=" << D;
    std::string msg = ss.str();

    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, msg);
  }

  const int n = gsl::narrow_cast<int>(N);
  const int d = gsl::narrow_cast<int>(D);
  const int nd = gsl::narrow_cast<int>(N * D);

  int num_split = 1;
  // Exponentiation
  #if defined USE_OPENMP
  const int max_split = 48;
  const int min_split_size = 16384;
  int min_row_per_split = (min_split_size + d - 1) / d;
  num_split = std::min((n + min_row_per_split - 1) / min_row_per_split, max_split);
  int rows_per_split = n / num_split;
  #endif

  if (num_split > 1) {
    #pragma omp parallel for
    for (int split = 0; split < num_split; ++split) {
      int start_row = split * rows_per_split;
      int real_row_count = (split < num_split - 1) ? rows_per_split : (n - start_row);
      int split_start = start_row * d;
      math::RowwiseMax<float, CPUMathUtil>(real_row_count, d, Xdata + split_start, rowmax + start_row, nullptr);
    }
  }
  else {
    math::RowwiseMax<float, CPUMathUtil>(n, d, Xdata, rowmax, nullptr);
  }


  if (num_split > 1) {
    #pragma omp parallel for
    for (int split = 0; split < num_split; ++split) {
      int start_row = split * rows_per_split;
      int real_row_count = (split < num_split - 1) ? rows_per_split : (n - start_row);
      int split_start = start_row * d;
      int real_elem_count = real_row_count * d;
      gsl::copy(gsl::make_span(Xdata + split_start, real_elem_count), gsl::make_span(Ydata + split_start, real_elem_count));
    }
  }
  else {
    // Put the intermediate result X - max(X) into Y by first copying X to Y, and then subtracting max from each entry
    gsl::copy(gsl::make_span(Xdata, nd), gsl::make_span(Ydata, nd));
  }

  math::Gemm<float, CPUMathUtil>(CblasNoTrans, CblasNoTrans, n, d, 1, -1, rowmax, sum_multiplier, 1, Ydata, nullptr);

  // Exponentiation
  //math::Exp<float, CPUMathUtil>(nd, Ydata, Ydata, nullptr);

  if (num_split > 1) {
    #pragma omp parallel for
    for (int split = 0; split < num_split; ++split) {
      int start_row = split * rows_per_split;
      int real_row_count = (split < num_split - 1) ? rows_per_split : (n - start_row);
      int split_start = start_row * d;
      int real_elem_count = real_row_count * d;
      math::Exp<float, CPUMathUtil>(real_elem_count, Ydata + split_start, Ydata + split_start, nullptr);
      //MlasComputeExpf(Ydata + split_start, Ydata + split_start, real_elem_count);
    }
  }
  else {
    math::Exp<float, CPUMathUtil>(nd, Ydata, Ydata, nullptr);
    //MlasComputeExpf(Ydata, Ydata, nd);
  }

  math::Gemv<float, CPUMathUtil>(CblasNoTrans, n, d, 1, Ydata, sum_multiplier, 0, scale, nullptr);

  // Do division
  if (!logarithmic) {
    if (num_split > 1) {
      #pragma omp parallel for
      for (int split = 0; split < num_split; ++split) {
        int start_row = split * rows_per_split;
        int real_row_count = (split < num_split - 1) ? rows_per_split : (n - start_row);
        for (int i = start_row, e = start_row + real_row_count; i < e; ++i) {
          for (int j = 0; j < D; ++j) {
            Ydata[i * D + j] /= scale[i];
          }
        }
      }
    }
    else {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
          Ydata[i * D + j] /= scale[i];
        }
      }
    }
  } else {
    if (num_split > 1) {
      #pragma omp parallel for
      for (int split = 0; split < num_split; ++split) {
        int start_row = split * rows_per_split;
        int real_row_count = (split < num_split - 1) ? rows_per_split : (n - start_row);
        for (int i = start_row, e = start_row + real_row_count; i < e; ++i) {
          for (int j = 0; j < D; ++j) {
            Ydata[i * D + j] = Xdata[i * D + j] - rowmax[i] - log(fmaxf(scale[i], 1e-20f));
          }
        }
      }
    }
    else {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
          Ydata[i * D + j] = Xdata[i * D + j] - rowmax[i] - log(fmaxf(scale[i], 1e-20f));
        }
      }
    }
  }

  return Status::OK();
}

//  math::Gemv<float, CPUMathUtil>(CblasNoTrans, n, d, 1, Ydata, sum_multiplier, 0, scale, nullptr);
//
//  // Do division
//  if (!logarithmic) {
//    for (int i = 0; i < N; ++i) {
//      for (int j = 0; j < D; ++j) {
//        Ydata[i * D + j] /= scale[i];
//      }
//    }
//  } else {
//    for (int i = 0; i < N; ++i) {
//      for (int j = 0; j < D; ++j) {
//        Ydata[i * D + j] = Xdata[i * D + j] - rowmax[i] - log(fmaxf(scale[i], 1e-20f));
//      }
//    }
//  }
//
//  return Status::OK();
//}
}  // namespace onnxruntime
