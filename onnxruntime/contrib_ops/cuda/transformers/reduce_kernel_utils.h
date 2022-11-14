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

template <typename T>
struct KeyValue {
  __device__ KeyValue() {
    key = -1;
    value = NumericLimits<T>::Min();
  }

  __device__ KeyValue(T value, int32_t key) : value(value), key(key) {
  }

  int32_t key;
  T value;
};

template <typename T>
__device__ __forceinline__ bool operator<(const KeyValue<T>& lh, const KeyValue<T>& rh) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
  return (float)lh.value < (float)rh.value || ((float)lh.value == (float)rh.value && rh.key != -1 && (lh.key == -1 || lh.key > rh.key));
#else
  return lh.value < rh.value || (lh.value == rh.value && rh.key != -1 && (lh.key == -1 || lh.key > rh.key));
#endif
}

template <typename T>
__device__ __forceinline__ bool Swap(T& lh, T& rh) {
  T tmp = rh;
  rh = lh;
  lh = tmp;
}

template <typename T, int max_k>
struct TopK {
  T elements[max_k];

  __device__ __forceinline__ void ShiftDown(int elem_size) {
    int cur_pos = 0;
    while (cur_pos < elem_size) {
      int32_t left_child_pos = 2 * cur_pos + 1;
      int32_t right_child_pos = 2 * cur_pos + 2;
      bool larger_than_left = left_child_pos < elem_size && elements[left_child_pos] < elements[cur_pos];
      bool larger_than_right = right_child_pos < elem_size && elements[right_child_pos] < elements[cur_pos];
      if (larger_than_left && larger_than_right) {
        if (elements[right_child_pos] < elements[left_child_pos]) {
          Swap(elements[cur_pos], elements[right_child_pos]);
          cur_pos = right_child_pos;
        } else {
          Swap(elements[cur_pos], elements[left_child_pos]);
          cur_pos = left_child_pos;
        }
      } else if (larger_than_left) {
        Swap(elements[cur_pos], elements[left_child_pos]);
        cur_pos = left_child_pos;
      } else if (larger_than_right) {
        Swap(elements[cur_pos], elements[right_child_pos]);
        cur_pos = right_child_pos;
      } else {
        break;
      }
    }
  }

  __device__ __forceinline__ void insert(T elem) {
    if (elements[0] < elem) {
      elements[0] = elem;
      ShiftDown(max_k);
    }
  }

  __device__ __forceinline__ void Sort() {
    for (int i = max_k - 1; i > 0; i--) {
      Swap(elements[0], elements[i]);
      ShiftDown(i);
    }
  }
};

template <typename T, int max_k>
__device__ __forceinline__ TopK<T, max_k> reduce_topk_op(const TopK<T, max_k>& a, const TopK<T, max_k>& b) {
  TopK<T, max_k> res = a;
  for (int i = 0; i < max_k; ++i)
    res.insert(b.elements[i]);
  return res;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
