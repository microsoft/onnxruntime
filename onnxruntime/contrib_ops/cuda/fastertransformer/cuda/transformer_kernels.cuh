/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "contrib_ops/cuda/fastertransformer/utils/arguments.h"
#include <assert.h>
#include "contrib_ops/cuda/fastertransformer/cuda/cuda_kernels.h"

namespace fastertransformer
{

template <typename T>
void add_bias_act_kernelLauncher(T* out, const T* bias, int m, int n, ActivationType activation_type, cudaStream_t stream);

template <typename T>
void add_bias_input_layernorm_kernelLauncher(T *out, const T *input_tensor,
                                             const T *bias, const T *gamma,
                                             const T *beta, int m, int n,
                                             cudaStream_t stream);

template <typename T>
void add_bias_input_layernorm_2_kernelLauncher(const T *from_tensor, const T *gamma,
                                               const T *beta, const T *bias,
                                               T *output, T *norm_output_buf_,
                                               const int m, const int n, cudaStream_t stream);

template <typename T>
void add_bias_input_kernelLauncher(T *output, const T *bias, const T *input, const int m, const int n, cudaStream_t stream);

template <typename T>
void layer_norm(const T *from_tensor, const T *gamma,
                const T *beta, T *norm_from_tensor_buf_, const int m, const int n, cudaStream_t stream);

} // namespace fastertransformer