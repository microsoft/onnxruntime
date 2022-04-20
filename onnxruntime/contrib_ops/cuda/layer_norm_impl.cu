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

//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// NVIDIA/apex is licensed under the
// BSD 3 - Clause "New" or "Revised" License
//

/* Modifications Copyright (c) Microsoft. */

#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/layer_norm_welford.cuh"
#include "layer_norm_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template<typename T, typename U, typename V, bool do_scale, bool do_center, bool simplified>
void LaunchLayerNorm(cudaStream_t stream, const int64_t num_instances, const int64_t norm_size,
                         const double epsilon, const T* input,U* mean, U* inv_variance,
                         const V* gamma, const V* beta, V* output) {
  using ComputeType = typename contrib::cuda::DefaultComputeType<U>::type;
  DirectLoad<T, ComputeType> load(input, norm_size);
  AffineStore<ComputeType, V, do_scale, do_center, simplified> store(output, norm_size, gamma, beta);
  DispatchLayerNorm<decltype(load), decltype(store), ComputeType, simplified>(
      stream, load, store, num_instances, norm_size, epsilon, mean, inv_variance);
}

template <typename T, typename U, typename V, bool simplified>
void ComputeLayerNorm(cudaStream_t stream, const int64_t num_instances, const int64_t norm_size,
                         const double epsilon, const T* input, U* mean, U* inv_variance,
                         const V* gamma, const V* beta, V* output) {
  if (gamma != nullptr && beta != nullptr) {
    LaunchLayerNorm<T, U, V, true, true, simplified>(stream, num_instances, norm_size, epsilon, input, mean, inv_variance,
                                        gamma, beta, output);
  } else if (gamma != nullptr && beta == nullptr) {
    LaunchLayerNorm<T, U, V, true, false, simplified>(stream, num_instances, norm_size, epsilon, input, mean, inv_variance,
                                        gamma, beta, output);
  } else if (gamma == nullptr && beta != nullptr) {
    LaunchLayerNorm<T, U, V, false, true, simplified>(stream, num_instances, norm_size, epsilon, input, mean, inv_variance,
                                        gamma, beta, output);
  } else {
    LaunchLayerNorm<T, U, V, false, false, simplified>(stream, num_instances, norm_size, epsilon, input, mean, inv_variance,
                                        gamma, beta, output);
  }
}

template <typename T, typename U, typename V, bool simplified>
void HostApplyLayerNorm(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    V* output,
    U* mean,
    U* inv_std_dev,
    const T* input,
    int n1,
    int n2,
    double epsilon,
    const V* gamma,
    const V* beta) {
  ComputeLayerNorm<T, U, V, simplified>(stream, n1, n2, epsilon, input, mean, inv_std_dev, gamma, beta, output);
}

#define LAYERNORM_LINEAR_IMPL(T, U, V, simplified)                                                                  \
  template void HostApplyLayerNorm<T, U, V, simplified>(const cudaDeviceProp& prop, cudaStream_t stream, V* output, \
                                                        U* mean, U* inv_std_dev, const T* input, int n1, int n2,    \
                                                        double epsilon, const V* gamma, const V* beta);

LAYERNORM_LINEAR_IMPL(float, float, float, true)
LAYERNORM_LINEAR_IMPL(half, float, half, true)
LAYERNORM_LINEAR_IMPL(double, double, double, true)
LAYERNORM_LINEAR_IMPL(float, float, half, true)
LAYERNORM_LINEAR_IMPL(float, float, float, false)
LAYERNORM_LINEAR_IMPL(half, float, half, false)
LAYERNORM_LINEAR_IMPL(double, double, double, false)
LAYERNORM_LINEAR_IMPL(float, float, half, false)
LAYERNORM_LINEAR_IMPL(BFloat16, float, BFloat16, true)
LAYERNORM_LINEAR_IMPL(BFloat16, float, BFloat16, false)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
