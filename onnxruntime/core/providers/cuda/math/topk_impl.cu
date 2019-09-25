// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void TopKKernel(const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t N, int64_t dimension) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  struct Node {
    T value_;
    int64_t index_;
    __device__ bool ValueEquals(const Node& node) const { return abs((float)value_ - (float)node.value_) < 1e-5; }
  };

  /* Implement a heap on deivcec to:
     1.Only keep fixed number of max/min nodes ever pushed
     2.Resort by new max/min policy
     3.Resort by value/index of the Node */
  class Heap {
   public:
    __device__ Heap(int64_t K, int64_t largest) : K_(K), largest_(largest) {
      nodes_ = new Node[K];
    }
    __device__ ~Heap() {
      delete[] nodes_;
    }
    __device__ const Node& Top() const { return nodes_[0]; }
    __device__ bool Allow(const Node& node) const {
      if (0 == K_)
        return false;
      else if (0 == size_)
        return true;
      else
        return ShouldGoTop(nodes_[0], node);
    }
    __device__ void Push(const Node& node) {
      if (size_ < K_) {
        nodes_[size_++] = node;
        SortFromBottom();
      } else if (Allow(node)) {
        nodes_[0] = node;
        SortFromTop();
      }
    }
    __device__ void Pop() {
      if (0 == size_) return;
      nodes_[0] = nodes_[--size_];
      SortFromTop();
    }
    __device__ bool Empty() const { return 0 == size_; }
    __device__ int64_t Size() const { return size_; }
    __device__ void Sort(int64_t largest = 1) {
      largest_ = largest;
      for (int64_t i = 1; i < size_; ++i) {
        SortFromBottom(i);
      }
    }
    bool sort_by_value_ = true; // on true sort by value then index, on false only sort by index
   private:
    __device__ bool ShouldGoTop(const Node& n1, const Node& n2) const {
      if (largest_ == 1) {
        return sort_by_value_ ? (n1.value_ < n2.value_ || n1.ValueEquals(n2) && n1.index_ > n2.index_) : n1.index_ < n2.index_;
      } else {
        return sort_by_value_ ? (n2.value_ < n1.value_ || n1.ValueEquals(n2) && n1.index_ > n2.index_) : n2.index_ < n1.index_;
      }
    }
    __device__ void SortFromTop() {
      int64_t pos = 0;
      int64_t rcd = pos + 1 << 1;
      int64_t lcd = rcd - 1;
      while (pos < size_) {
        if (lcd >= size_) break;
        auto nxt = rcd >= size_ ? lcd : ShouldGoTop(nodes_[rcd], nodes_[lcd]) ? rcd : lcd;
        if (ShouldGoTop(nodes_[nxt], nodes_[pos])) {
          auto tmp_nd = nodes_[pos];
          nodes_[pos] = nodes_[nxt];
          nodes_[nxt] = tmp_nd;
          pos = nxt;
          rcd = pos + 1 << 1;
          lcd = rcd - 1;
        } else
          break;
      }
    }
    __device__ void SortFromBottom(int64_t pos = -1) {
      if (pos == -1) {
        pos = size_ - 1;
      }
      auto pap = pos - 1 >> 1;
      while (pos > 0 && ShouldGoTop(nodes_[pos], nodes_[pap])) {
        auto tmp_nd = nodes_[pos];
        nodes_[pos] = nodes_[pap];
        nodes_[pap] = tmp_nd;
        pos = pap;
        pap = pos - 1 >> 1;
      }
    }
    int64_t K_;
    int64_t size_ = 0;
    int64_t largest_ = 1; // keep largest K ever pushed with min on top, or keep smallest K with max on top 
    Node* nodes_ = nullptr;
  };

  Heap heap(K, largest);
  auto left = id / (axis == size - 1 ? 1 : elem_nums[axis + 1]) * elem_nums[axis];
  auto right = axis == size - 1 ? 0 : id % elem_nums[axis + 1];
  for (int64_t i = 0; i < dimension; ++i) {
    auto input_offset = left + i * (axis == size - 1 ? 1 : elem_nums[axis + 1]) + right;
    heap.Push({input_x[input_offset], i});
  }
  if (1 != sorted) {
    heap.sort_by_value_ = false;
    heap.Sort(-1);
  }
  left = left * K / dimension;
  while (!heap.Empty()) {
    auto& node = heap.Top();
    auto output_offset = left + (heap.Size() - 1) * (axis == size - 1 ? 1 : elem_nums[axis + 1]) + right;
    output_v[output_offset] = node.value_;
    output_i[output_offset] = node.index_;
    heap.Pop();
  }
}

template <typename T>
Status TopKImpl(const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t N, int64_t dimension) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  TopKKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(input_x, output_v, output_i, elem_nums, size, axis, K, largest, sorted, N, dimension);
  return Status::OK();
}

#define TOPKIMPLE(T) template Status TopKImpl<T>(const T* input_x,         \
                                                 T* output_v,              \
                                                 int64_t* output_i,        \
                                                 const int64_t* elem_nums, \
                                                 size_t size,              \
                                                 int64_t axis,             \
                                                 int64_t K,                \
                                                 int64_t largest,          \
                                                 int64_t sorted,           \
                                                 int64_t N,                \
                                                 int64_t dimension)

TOPKIMPLE(uint8_t);
TOPKIMPLE(uint16_t);
TOPKIMPLE(uint32_t);
TOPKIMPLE(uint64_t);
TOPKIMPLE(int8_t);
TOPKIMPLE(int16_t);
TOPKIMPLE(int32_t);
TOPKIMPLE(int64_t);
TOPKIMPLE(float);
TOPKIMPLE(double);

}  // namespace cuda
}  // namespace onnxruntime