// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/ort_mutex.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include <list>

namespace onnxruntime {
namespace cuda {

class CudnnConvolutionDescriptor final {
 public:
  CudnnConvolutionDescriptor();
  ~CudnnConvolutionDescriptor();

  Status Set(size_t rank,
             const std::vector<int64_t>& pads,
             const std::vector<int64_t>& strides,
             const std::vector<int64_t>& dilations,
             int groups,
             cudnnConvolutionMode_t mode,
             cudnnDataType_t data_type);

  operator cudnnConvolutionDescriptor_t() const { return desc_; }

 private:
  cudnnConvolutionDescriptor_t desc_;
};

template <typename T>
struct vector_hash {
  std::size_t operator()(const std::vector<T>& values) const {
    std::size_t seed = values.size();
    for (auto& val : values)
      seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

template <typename Key, typename T,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>,
          typename ListAllocator = std::allocator<Key>>
class lru_unordered_map {
 public:
  lru_unordered_map(size_t max_size) : max_size_(max_size) {}

  void insert(const Key& key, const T& value) {
    auto it = items_.find(key);
    if (it != items_.end()) {
      it->second.value = value;
      move_to_front(it->second.lru_iterator);
      return;
    }

    while (size() + 1 > max_size_) {
      items_.erase(lru_list_.back());
      lru_list_.pop_back();
    }

    lru_list_.emplace_front(key);
    items_.emplace(key, value_type{value, lru_list_.begin()});
  }

  T& at(const Key& key) {
    auto it = items_.find(key);
    if (it == items_.end()) {
      throw std::out_of_range("There is no such key in cache");
    }
    move_to_front(it->second.lru_iterator);
    return it->second.value;
  }

  bool contains(const Key& key) const {
    return items_.find(key) != items_.end();
  }

  size_t size() const {
    return items_.size();
  }

  void clear() {
    items_.clear();
    lru_list_.clear();
  }

 private:
  using list_type = std::list<Key, ListAllocator>;
  using iterator_type = typename list_type::iterator;
  struct value_type {
    T value;
    iterator_type lru_iterator;
  };
  using MapAllocator = std::allocator<std::pair<const Key, value_type>>;

  void move_to_front(iterator_type it) {
    lru_list_.splice(lru_list_.begin(), lru_list_, it);
  }

  size_t max_size_;
  std::unordered_map<Key, value_type, Hash, KeyEqual, MapAllocator> items_;
  list_type lru_list_;
};

// cached cudnn descriptors
constexpr size_t MAX_CACHED_ALGO_PERF_RESULTS = 10000;

template <typename AlgoPerfType>
struct CudnnConvState {
  // if x/w dims changed, update algo and cudnnTensors
  std::vector<int64_t> last_x_dims;
  std::vector<int64_t> last_w_dims;

  // these would be recomputed if x/w dims change
  std::vector<int64_t> y_dims;
  std::vector<int64_t> y_dims_with_adjusted_pads;
  size_t workspace_bytes;
  decltype(AlgoPerfType().algo) algo;
  CudnnTensor x_tensor;
  const void* x_data = nullptr;
  size_t element_size = 0;
  CudnnFilterDescriptor w_desc;
  const void* w_data = nullptr;
  CudnnTensor b_tensor;
  const void* b_data = nullptr;
  void* b_zero = nullptr;
  CudnnTensor y_tensor;
  Tensor* Y = nullptr;
  void* y_data = nullptr;
  CudnnTensor z_tensor;
  const void* z_data = nullptr;
  CudnnConvolutionDescriptor conv_desc;

  struct PerfResultParams {
    decltype(AlgoPerfType().algo) algo;
    decltype(AlgoPerfType().memory) memory;
    decltype(AlgoPerfType().mathType) mathType;
  };

  lru_unordered_map<std::vector<int64_t>, PerfResultParams, vector_hash<int64_t>> cached_benchmark_results{MAX_CACHED_ALGO_PERF_RESULTS};

  // Some properties needed to support asymmetric padded Conv nodes
  bool post_slicing_required;
  std::vector<int64_t> slice_starts;
  std::vector<int64_t> slice_ends;
  std::vector<int64_t> slice_axes;

  // note that conv objects are shared between execution frames, and a lock is needed to avoid multi-thread racing
  OrtMutex mutex;
  IAllocatorUniquePtr<void> memory_for_cudnn_conv_results;

  ~CudnnConvState() {
    if (b_zero) {
      CUDA_CALL_THROW(cudaFree(b_zero));
      b_zero = nullptr;
    }
  }
};

enum : size_t {
  AlgoSearchWorkspaceSize = 32 * 1024 * 1024,
};

template <typename T>
class Conv : public CudaKernel {
 public:
  using CudaT = typename ToCudaType<T>::MappedType;

  Conv(const OpKernelInfo& info) : CudaKernel(info), conv_attrs_(info) {
    auto pads_size = conv_attrs_.pads.size();
    ORT_ENFORCE(pads_size % 2 == 0);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  inline IAllocatorUniquePtr<void> GetWorkSpace() const {
    return GetScratchBuffer<void>(s_.workspace_bytes);
  }

  Status UpdateState(OpKernelContext* context, bool bias_expected = false) const;
  ConvAttributes conv_attrs_;
  mutable CudnnConvState<cudnnConvolutionFwdAlgoPerf_t> s_;
  constexpr static auto kDefaultConvAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
};

Status SliceOutUnwantedOutputSection(cudaStream_t stream,
                                     const void* input_data,
                                     const std::vector<int64_t>& input_dims,
                                     void* output_data,
                                     const std::vector<int64_t>& output_dims,
                                     std::vector<int64_t> starts,
                                     const std::vector<int64_t>& ends,
                                     const std::vector<int64_t>& axes,
                                     size_t element_size);
}  // namespace cuda
}  // namespace onnxruntime
