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
/* Modifications Copyright (c) Microsoft. */
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <type_traits>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T, int MAX_K>
struct TopK {
  int32_t key[MAX_K];
  T value[MAX_K];

  __device__ __forceinline__ void insert(T elem, int elem_id) {
    T v = value[MAX_K - 1];
    if (v < elem ||
        (key[MAX_K - 1] == -1) || ((elem == value[MAX_K - 1]) && (elem_id < key[MAX_K - 1])))
    // if (elem > u[MAX_K-1] || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
    {
      value[MAX_K - 1] = elem;
      key[MAX_K - 1] = elem_id;
    }

    for (int k = MAX_K - 2; k >= 0; --k) {
      if ((value[k + 1] > value[k]) || (key[k] == -1) || ((value[k + 1] == value[k]) && (key[k + 1] < key[k])))
      // if ((u[k+1] > u[k]) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
      {
        T u2 = value[k];
        int p2 = key[k];
        value[k] = value[k + 1];
        key[k] = key[k + 1];
        value[k + 1] = u2;
        key[k + 1] = p2;
      }
    }
  }

  __device__ __forceinline__ void init() {
    for (int i = 0; i < MAX_K; i++) {
      key[i] = -1;
      value[i] = NumericLimits<T>::Min();
    }
  }
};

template <typename T, int MAX_K>
__device__ __forceinline__ TopK<T, MAX_K> reduce_topk_op(const TopK<T, MAX_K>& a, const TopK<T, MAX_K>& b) {
  TopK<T, MAX_K> res = a;
  for (int i = 0; i < MAX_K; ++i)
    res.insert(b.value[i], b.key[i]);
  return res;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
