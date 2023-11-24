// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/ort_mutex.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include <list>

namespace onnxruntime {

using ConvPadVector = ConvAttributes::ConvPadVector;

namespace rocm {

class MiopenConvolutionDescriptor final {
 public:
  MiopenConvolutionDescriptor();
  ~MiopenConvolutionDescriptor();

  Status Set(size_t rank,
             const gsl::span<const int64_t>& pads,
             const gsl::span<const int64_t>& strides,
             const gsl::span<const int64_t>& dilations,
             int groups,
             miopenConvolutionMode_t mode,
             miopenDataType_t data_type);

  operator miopenConvolutionDescriptor_t() const { return desc_; }

 private:
  miopenConvolutionDescriptor_t desc_;
};

struct vector_hash {
  std::size_t operator()(const TensorShapeVector& values) const {
    std::size_t seed = values.size();
    for (auto& val : values)
      seed ^= std::hash<int64_t>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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

// cached miopen descriptors
constexpr size_t MAX_CACHED_ALGO_PERF_RESULTS = 10000;

template <typename AlgoPerfType>
struct MiopenConvState {
  // if x/w dims changed, update algo and miopenTensors
  TensorShape last_x_dims;
  TensorShape last_w_dims;

  // these would be recomputed if x/w dims change
  TensorShape y_dims;
  TensorShapeVector y_dims_with_adjusted_pads;
  size_t workspace_bytes;
  decltype(AlgoPerfType().bwd_data_algo) bwd_data_algo;
  decltype(AlgoPerfType().fwd_algo) fwd_algo;
  MiopenTensor x_tensor;
  const void* x_data = nullptr;
  size_t element_size = 0;
  MiopenTensorDescriptor w_desc;
  const void* w_data = nullptr;
  MiopenTensor b_tensor;
  const void* b_data = nullptr;
  void* b_zero = nullptr;
  MiopenTensor y_tensor;
  Tensor* Y = nullptr;
  void* y_data = nullptr;
  MiopenTensor z_tensor;
  const void* z_data = nullptr;
  MiopenConvolutionDescriptor conv_desc;

  struct PerfFwdResultParams {
    decltype(AlgoPerfType().fwd_algo) fwd_algo;
    decltype(AlgoPerfType().memory) memory;
  };

  struct PerfBwdResultParams {
    decltype(AlgoPerfType().bwd_data_algo) bwd_data_algo;
    decltype(AlgoPerfType().memory) memory;
  };

  lru_unordered_map<TensorShapeVector, PerfFwdResultParams, vector_hash> cached_benchmark_fwd_results{MAX_CACHED_ALGO_PERF_RESULTS};
  lru_unordered_map<TensorShapeVector, PerfBwdResultParams, vector_hash> cached_benchmark_bwd_results{MAX_CACHED_ALGO_PERF_RESULTS};

  // Some properties needed to support asymmetric padded Conv nodes
  bool post_slicing_required;
  TensorShapeVector slice_starts;
  TensorShapeVector slice_ends;
  TensorShapeVector slice_axes;

  // note that conv objects are shared between execution frames, and a lock is needed to avoid multi-thread racing
  OrtMutex mutex;
  IAllocatorUniquePtr<void> memory_for_miopen_conv_results;

  ~MiopenConvState() {
    if (b_zero) {
      HIP_CALL_THROW(hipFree(b_zero));
      b_zero = nullptr;
    }
  }
};

enum : size_t {
  AlgoSearchWorkspaceSize = 32 * 1024 * 1024,
};

// ONNX Conv operator uses NCHW format for input, weights and output.
// NhwcConv contrib ops uses NHWC format: last dimension of input, weights and output are channels.
template <typename T, bool NHWC>
class Conv : public RocmKernel {
 public:
  using HipT = typename ToHipType<T>::MappedType;

  Conv(const OpKernelInfo& info) : RocmKernel(info), conv_attrs_(info) {
    auto pads_size = conv_attrs_.pads.size();
    ORT_ENFORCE(pads_size % 2 == 0);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  inline IAllocatorUniquePtr<void> GetWorkSpace(onnxruntime::Stream* stream) const {
    return GetScratchBuffer<void>(s_.workspace_bytes, stream);
  }

  Status UpdateState(OpKernelContext* context, bool bias_expected = false) const;
  ConvAttributes conv_attrs_;
  mutable MiopenConvState<miopenConvAlgoPerf_t> s_;
  constexpr static auto kDefaultConvAlgo = miopenConvolutionFwdAlgoGEMM;
  static const miopenConvFwdAlgorithm_t kAllAlgos[];
};

Status SliceOutUnwantedOutputSection(hipStream_t stream,
                                     const void* input_data,
                                     gsl::span<const int64_t> input_dims,
                                     void* output_data,
                                     const gsl::span<const int64_t>& output_dims,
                                     const gsl::span<const int64_t>& starts,
                                     const gsl::span<const int64_t>& ends,
                                     const gsl::span<const int64_t>& axes,
                                     size_t element_size);
}  // namespace rocm
}  // namespace onnxruntime
