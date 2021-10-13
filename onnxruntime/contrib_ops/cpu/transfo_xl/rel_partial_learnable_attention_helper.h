// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

template <typename T>
void ComputeRelPartialLearnableAttentionSoftmaxInplace(T* score, int N, int D, ThreadPool* tp) {
  ThreadPool::TryParallelFor(tp, N, D * 2.0, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    for (std::ptrdiff_t j = begin; j != end; ++j) {
      float* x = reinterpret_cast<T*>(score) + j * D;
      float* y = x;

      // e^x is represented as infinity if x is large enough, like 100.f.
      // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if
      // one or more item are large enough. a math transform as below is
      // leveraged to get a stable softmax: e^xi/(e^x1 + ...e^xn) = e^(xi -
      // max) / (e^(x1 - max) + ... + e^(xn - max))
      float max = -std::numeric_limits<float>::infinity();
      for (int i = 0; i < D; i++) {
        if (max < x[i])
          max = x[i];
      }
      for (int i = 0; i < D; i++) {
        y[i] = expf(x[i] - max);
      }

      double sum = 0.0;

      for (int i = 0; i < D; i++) {
        sum += x[i];
      }

      if (sum == 0) {
        for (int i = 0; i < D; i++) {
          y[i] = 1.0f / (float)D;
        }
      } else {
        for (int i = 0; i < D; i++) {
          y[i] = x[i] / (float)sum;
        }
      }
    }
  });
}

template <>
inline void ComputeRelPartialLearnableAttentionSoftmaxInplace(float* score, int N, int D, ThreadPool* tp) {
  MlasComputeSoftmax(score, score, N, D, false, tp);
}

}  // namespace contrib
}  // namespace onnxruntime
