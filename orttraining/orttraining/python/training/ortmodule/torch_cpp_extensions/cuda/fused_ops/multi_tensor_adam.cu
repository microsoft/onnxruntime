// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is adapted from microsoft/DeepSpeed
// multi_tensor_adam.cu

/* Copyright 2020 The Microsoft DeepSpeed Team
   Copyright NVIDIA/apex
   This file is adapted from fused adam in NVIDIA/apex, commit a109f85
*/

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "multi_tensor_apply.cuh"
#include "type_shim.h"
#include <cmath>

#define BLOCK_SIZE 512
#define ILP 4

typedef enum {
    ADAM_MODE_0 = 0,  // L2 regularization mode
    ADAM_MODE_1 = 1   // Decoupled weight decay mode(AdamW)
} adamMode_t;

using MATH_T = float;

template <typename T>
struct AdamFunctor {
    __device__ __forceinline__ void operator()(int chunk_size,
                                               volatile int* noop_gmem,
                                               TensorListMetadata<4>& tl,
                                               const float beta1,
                                               const float beta2,
                                               const float epsilon,
                                               const float lr,
                                               const float lr_corrected,
                                               adamMode_t mode,
                                               const float decay)
    {
        int tensor_loc = tl.block_to_tensor[blockIdx.x];

        int chunk_idx = tl.block_to_chunk[blockIdx.x];
        int n = tl.sizes[tensor_loc];

        T* g = (T*)tl.addresses[0][tensor_loc];
        g += chunk_idx * chunk_size;

        T* p = (T*)tl.addresses[1][tensor_loc];
        p += chunk_idx * chunk_size;

        T* m = (T*)tl.addresses[2][tensor_loc];
        m += chunk_idx * chunk_size;

        T* v = (T*)tl.addresses[3][tensor_loc];
        v += chunk_idx * chunk_size;

        n -= chunk_idx * chunk_size;

        // see note in multi_tensor_scale_kernel.cu
        for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
            MATH_T r_g[ILP];
            MATH_T r_p[ILP];
            MATH_T r_m[ILP];
            MATH_T r_v[ILP];
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                int i = i_start + threadIdx.x + ii * blockDim.x;
                if (i < n && i < chunk_size) {
                    r_g[ii] = g[i];
                    r_p[ii] = p[i];
                    r_m[ii] = m[i];
                    r_v[ii] = v[i];
                } else {
                    r_g[ii] = MATH_T(0);
                    r_p[ii] = MATH_T(0);
                    r_m[ii] = MATH_T(0);
                    r_v[ii] = MATH_T(0);
                }
            }
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                if (mode == ADAM_MODE_0) {  // L2
                    r_g[ii] = r_g[ii] + (decay * r_p[ii]);
                    r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
                    r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
                    MATH_T denom = sqrtf(r_v[ii]) + epsilon;
                    r_p[ii] = r_p[ii] - (lr_corrected * r_m[ii] / denom);
                } else {  // weight decay
                    // Adapted to be mathematically equivalent to transformers AdamW
                    r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
                    r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
                    MATH_T denom = sqrtf(r_v[ii]) + epsilon;
                    r_p[ii] = r_p[ii] - (lr_corrected * r_m[ii] / denom);
                    r_p[ii] = r_p[ii] - (lr * decay * r_p[ii]);
                }
            }
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                int i = i_start + threadIdx.x + ii * blockDim.x;
                if (i < n && i < chunk_size) {
                    p[i] = r_p[ii];
                    m[i] = r_m[ii];
                    v[i] = r_v[ii];
                }
            }
        }
    }
};

void multi_tensor_adam_cuda(int chunk_size,
                            at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr,
                            const float beta1,
                            const float beta2,
                            const float epsilon,
                            const int step,
                            const int mode,
                            const int bias_correction,
                            const float weight_decay)
{
    using namespace at;

    // Handle bias correction mode
    double bias_correction1 = 1.0, bias_correction2 = 1.0;
    float lr_corrected = lr;
    if (bias_correction == 1) {
        bias_correction1 = 1 - std::pow(beta1, step);
        bias_correction2 = 1 - std::pow(beta2, step);
        lr_corrected *= std::sqrt(bias_correction2) / bias_correction1;
    }

    // Assume single type across p,g,m1,m2 now
    DISPATCH_DOUBLE_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(),
                                   0,
                                   "adam",
                                   multi_tensor_apply<4>(BLOCK_SIZE,
                                                         chunk_size,
                                                         noop_flag,
                                                         tensor_lists,
                                                         AdamFunctor<scalar_t_0>(),
                                                         beta1,
                                                         beta2,
                                                         epsilon,
                                                         lr,
                                                         lr_corrected,
                                                         (adamMode_t)mode,
                                                         weight_decay);)

    AT_CUDA_CHECK(cudaGetLastError());
}
