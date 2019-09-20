// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void TopKKernel(const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t N, int64_t dimension)
{
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  struct Node {
    T       value_;
    int64_t index_;
  };

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
      if (size_ == 0) return true;
      return ShouldGoTop(nodes_[0], node);
    }
    __device__ void Push (const Node& node) {
      if (size_ != K_) {
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
    __device__ bool Empty() const { return 0 == size_;}
    __device__ void Sort(int64_t largest = 1) {
      largest_ = largest;
      for (int64_t i = 1; i < size_; ++i) {
        SortFromBottom(i);
      }
    }
    bool sort_by_value_ = true;
  private:
    __device__ bool ShouldGoTop(const Node& n1, const Node& n2) const {
      if (largest_ == 1) return sort_by_value_ ? n1.value_ < n2.value_ : n1.index_ < n2.index_;
      else               return sort_by_value_ ? n2.value_ < n1.value_ : n2.index_ < n1.index_;
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
        } else break;
      }
    }
    __device__ void SortFromBottom(int64_t pos = -1) {
      if (pos == -1) {
        pos = size_ - 1;
      }
      auto pap = pos >> 1;
      while (pos > 0 && ShouldGoTop(nodes_[pos], nodes_[pap])) {
        auto tmp_nd = nodes_[pos];
        nodes_[pos] = nodes_[pap];
        nodes_[pap] = tmp_nd;
        pos = pap;
        pap = pos >> 1;
      }
    }
    int64_t K_;
    int64_t size_    = 0;
    int64_t largest_ = 1; // keep largest K ever pushed
    Node*   nodes_   = nullptr;
  };

  Heap heap(K, largest);
  auto left = N / (axis == size-1 ? 1 : elem_nums[axis+1]) * elem_nums[axis];
  auto right = axis == size-1 ? 0 : N % elem_nums[axis+1];
  for (int64_t i = 0; i < dimension; ++i) {
    auto input_offset = left + i * (axis == size-1 ? 1 : elem_nums[axis+1]) + right;
    heap.Push({input_x[input_offset], i});
  }
  if (sorted != 1) {
    heap.sort_by_value_ = false;
    heap.Sort(); //sort heap by ascending index
  }
  int64_t output_count = 0;
  while (!heap.Empty()) {
    auto& node = heap.Top();
    auto output_offset = left + output_count * (axis == size-1 ? 1 : elem_nums[axis+1]) + right;
    output_v[output_offset] = node.value_;
    output_i[output_offset] = node.index_;
    output_count++;
    heap.Pop();
  }
}

template <typename T>
Status TopKImpl(const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted)
{
  auto dimension = axis == size-1 ? elem_nums[axis] : elem_nums[axis] / elem_nums[axis+1];
  auto N = elem_nums[0]/dimension;
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  TopKKernel <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>> (input_x, output_v, output_i, elem_nums, size, axis, K, largest, sorted, N, dimension);
  return Status::OK();
}

template Status TopKImpl<int32_t> (const int32_t* input_x, int32_t* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted);

}
}