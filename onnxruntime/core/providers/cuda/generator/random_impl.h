// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace cuda {

#define RANDOM_KERNEL_DECLARE(name)                                                                          \
  template <typename T>                                                                                      \
  void name##KernelImpl(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N, const float alpha, \
                        const float beta, PhiloxGenerator& generator, T* Y_data);

RANDOM_KERNEL_DECLARE(RandomNormal)
RANDOM_KERNEL_DECLARE(RandomUniform)

#undef RANDOM_KERNEL_DECLARE

}  // namespace cuda
}  // namespace onnxruntime
