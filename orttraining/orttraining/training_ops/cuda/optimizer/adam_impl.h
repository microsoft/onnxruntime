// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD, typename T_GRAD_NORM, typename T_MIXED_PRECISION_FP>
void AdamOptimizerImpl(
    cudaStream_t stream,
    const T1* eta,
    const T2 update_count,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    const T3* loss_scale,
    const T_GRAD_NORM* grad_norm,
    const float alpha,
    const float beta,
    const float lambda,
    const float epsilon,
    const float max_norm,
    const bool do_bias_correction,
    const int64_t weight_decay_mode,
    T4* moment_1_out,
    T4* moment_2_out,
    T3* weights_out,
    T_GRAD* grads_out,
    T_MIXED_PRECISION_FP* mixed_precision_weights_out,
    size_t count);
}
}  // namespace onnxruntime
