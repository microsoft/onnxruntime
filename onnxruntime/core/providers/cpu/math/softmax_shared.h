// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
/**
Calculate Softmax using CPU memory.
@param N Number of rows
@param D Number of elements in each row
@param Xdata Source data
@param Ydata Output data
@param scale Storage for scale calculation. Size must be >= N.
@param sum_multiplier Weights for each element. Size must be >= D.
@param logarithmic If true, compute LogSoftmax. If false compute Softmax.
@param rowmax Storage for calculation of maximum in each row. Size must be >= N.
*/
common::Status SoftmaxCPU(int64_t N, int64_t D, const float* Xdata, float* Ydata, float* scale,
                          const float* sum_multiplier, bool logarithmic, float* rowmax, concurrency::ThreadPool* tp);
}  // namespace onnxruntime
