// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <thread>
#include <iterator>
#include <queue>
#include <iostream>
#include <algorithm>
#include "core/platform/threadpool.h"

namespace onnxruntime {

void shape2strides(const TensorShape& shape, TensorShape& shape_strides) {
  std::vector<int64_t> strides;
  strides.resize(shape.NumDimensions());
  strides[strides.size() - 1] = sizeof(int64_t);
  for (int i = (int)strides.size() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  shape_strides = strides;
}

template <typename NTYPE>
struct HeapMax {
  using DataType = NTYPE;
  bool cmp1(const NTYPE& v1, const NTYPE& v2) const {
    return v1 > v2;
  }
  bool cmp(int64_t i1, int64_t i2, const NTYPE* ens, const int64_t* pos) const {
    return (ens[pos[i1]] < ens[pos[i2]]) || ((pos[i1] > pos[i2]) && (ens[pos[i1]] == ens[pos[i2]]));
  }
};

template <typename NTYPE>
struct HeapMin {
  using DataType = NTYPE;
  bool cmp1(const NTYPE& v1, const NTYPE& v2) const {
    return v1 < v2;
  }
  bool cmp(int64_t i1, int64_t i2, const NTYPE* ens, const int64_t* pos) const {
    return (ens[pos[i1]] > ens[pos[i2]]) || ((pos[i1] > pos[i2]) && (ens[pos[i1]] == ens[pos[i2]]));
  }
};

template <class HeapCmp>
void _heapify_up_position(const typename HeapCmp::DataType* ens, int64_t* pos, size_t i, size_t k, const HeapCmp& heap_cmp) {
  size_t left, right;
  int64_t ch;
  while (true) {
    left = 2 * i + 1;
    right = left + 1;
    if (right < k) {
      if (heap_cmp.cmp(left, i, ens, pos) && !heap_cmp.cmp1(ens[pos[left]], ens[pos[right]])) {
        ch = pos[i];
        pos[i] = pos[left];
        pos[left] = ch;
        i = left;
      } else if (heap_cmp.cmp(right, i, ens, pos)) {
        ch = pos[i];
        pos[i] = pos[right];
        pos[right] = ch;
        i = right;
      } else
        break;
    } else if ((left < k) && heap_cmp.cmp(left, i, ens, pos)) {
      ch = pos[i];
      pos[i] = pos[left];
      pos[left] = ch;
      i = left;
    } else
      break;
  }
}

template <class HeapCmp>
void _topk_element(const typename HeapCmp::DataType* values, size_t k, size_t n, int64_t* indices, bool sorted, const HeapCmp& heap_cmp) {
  if (n <= k && !sorted) {
    for (size_t i = 0; i < n; ++i, ++indices)
      *indices = i;
  } else if (k == 1) {
    auto begin = values;
    auto end = values + n;
    *indices = 0;
    for (++values; values != end; ++values)
      *indices = heap_cmp.cmp1(*values, begin[*indices]) ? ((int64_t)(values - begin)) : *indices;
  } else {
    indices[k - 1] = 0;

    size_t i = 0;
    for (; i < k; ++i) {
      indices[k - i - 1] = i;
      _heapify_up_position(values, indices, k - i - 1, k, heap_cmp);
    }
    for (; i < n; ++i) {
      if (heap_cmp.cmp1(values[i], values[indices[0]])) {
        indices[0] = i;
        _heapify_up_position(values, indices, 0, k, heap_cmp);
      }
    }
    if (sorted) {
      int64_t ech;
      i = k - 1;
      ech = indices[0];
      indices[0] = indices[i];
      indices[i] = ech;
      --i;
      for (; i > 0; --i) {
        _heapify_up_position(values, indices, 0, i + 1, heap_cmp);
        ech = indices[0];
        indices[0] = indices[i];
        indices[i] = ech;
      }
    }
  }
}

template <class HeapCmp>
void topk_element(concurrency::ThreadPool* thread_pool, int64_t* pos, size_t k, const typename HeapCmp::DataType* values, const TensorShape& shape, bool sorted, int64_t th_parallel) {
  HeapCmp heap_cmp;
  if (shape.NumDimensions() == 1) {
    _topk_element(values, k, shape[0], pos, sorted, heap_cmp);
  } else {
    auto vdim = shape[shape.NumDimensions() - 1];
    auto ptr = pos;

    if (shape[0] <= th_parallel) {
      auto tdim = shape.Size();
      const typename HeapCmp::DataType* data = values;
      const typename HeapCmp::DataType* end = data + tdim;
      for (; data != end; data += vdim, ptr += k)
        _topk_element(data, k, vdim, ptr, sorted, heap_cmp);
    } else {
      // parallelisation
      const typename HeapCmp::DataType* data = values;
      concurrency::ThreadPool::TryBatchParallelFor(
          thread_pool,
          static_cast<int32_t>(shape[0]),
          [&](ptrdiff_t nr) {
            _topk_element(data + nr * vdim, k, vdim, ptr + nr * k, sorted, heap_cmp);
          },
          0);
    }
  }
}

template <typename NTYPE>
void topk_element_fetch(NTYPE* ptr, const NTYPE* data_val, const TensorShape& shape_val, const int64_t* data_ind, const TensorShape& shape_ind) {
  auto tdim = shape_ind.Size();
  auto dim_val = shape_val[shape_val.NumDimensions() - 1];
  auto dim_ind = shape_ind[shape_ind.NumDimensions() - 1];
  const int64_t* end_ind = data_ind + tdim;

  const int64_t* next_end_ind;
  for (; data_ind != end_ind; data_val += dim_val) {
    next_end_ind = data_ind + dim_ind;
    for (; data_ind != next_end_ind; ++data_ind, ++ptr)
      *ptr = data_val[*data_ind];
  }
}

}  // namespace onnxruntime