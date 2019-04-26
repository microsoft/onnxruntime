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

  math::RowwiseMax<float, CPUMathUtil>(n, d, Xdata, rowmax, nullptr);

  // Put the intermediate result X - max(X) into Y by first copying X to Y, and then subtracting max from each entry
  gsl::copy(gsl::make_span(Xdata, nd), gsl::make_span(Ydata, nd));

  math::Gemm<float, CPUMathUtil>(CblasNoTrans, CblasNoTrans, n, d, 1, -1, rowmax, sum_multiplier, 1, Ydata, nullptr);

  // Exponentiation
  math::Exp<float, CPUMathUtil>(nd, Ydata, Ydata, nullptr);
  math::Gemv<float, CPUMathUtil>(CblasNoTrans, n, d, 1, Ydata, sum_multiplier, 0, scale, nullptr);

  // Do division
  if (!logarithmic) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        Ydata[i * D + j] /= scale[i];
      }
    }
  } else {
    for (int i = 0; i < N; ++i) {
      auto log_fmaxf_scale_i = log(fmaxf(scale[i], 1e-20f));
      for (int j = 0; j < D; ++j) {
        Ydata[i * D + j] = Xdata[i * D + j] - rowmax[i] - log_fmaxf_scale_i;
      }
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
