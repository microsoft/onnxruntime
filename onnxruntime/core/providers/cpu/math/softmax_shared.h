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
@param logarithmic If true, compute LogSoftmax. If false compute Softmax.
*/
template <typename T>
common::Status SoftmaxCPU(size_t N, size_t D, const T* Xdata, T* Ydata,
                          bool logarithmic, concurrency::ThreadPool* thread_pool);
}  // namespace onnxruntime
