// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "device_array.h"
#include "operator.h"
#include "contrib_ops/rocm/bert/fast_gelu_impl_kernel.h"

template <typename T, int ThreadsPerBlock, int VecSize>
void LaunchFastGelu(const T* input, const T* bias, T* output, int input_length, int bias_length) {
  hipLaunchKernelGGL((onnxruntime::contrib::rocm::FastGeluKernelVec<T, ThreadsPerBlock, VecSize>), 
                  dim3(ceil(float(input_length)/(float(ThreadsPerBlock)*VecSize))),
                  dim3(ThreadsPerBlock),
                  0, 0,
                  input_length, bias_length, input, bias, output);
}
