// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/conv_base.h"
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
             cudnnConvolutionMode_t mode,
             cudnnDataType_t data_type);

  operator cudnnConvolutionDescriptor_t() const { return desc_; }

 private:
  cudnnConvolutionDescriptor_t desc_;
};

template <typename Key, typename T>
class lru_map {
 public:
  lru_map(size_t max_size) : max_size_(max_size) {}

  void insert(const Key& key, const T& value) {
    auto it = handles_.find(key);
    if (it != handles_.end()) {
      items_.erase(it->second);
      handles_.erase(it);
    }

    items_.emplace_front(key, value);
    handles_.emplace(key, items_.begin());

    while (size() > max_size_) {
      handles_.erase(items_.back().first);
      items_.pop_back();
    }
  }

  T& at(const Key& key) {
    auto it = handles_.find(key);
    if (it == handles_.end()) {
      throw std::out_of_range("There is no such key in cache");
    }
    items_.splice(items_.begin(), items_, it->second);
    return it->second->second;
  }

  bool contains(const Key& key) const {
    return handles_.find(key) != handles_.end();
  }

  size_t size() const {
    return handles_.size();
  }

  void clear() {
    items_.clear();
    handles_.clear();
  }

private:
  using value_type = std::pair<Key, T>;
  using iterator_type = typename std::list<value_type>::iterator;

  size_t max_size_;
  std::list<value_type> items_;
  std::map<Key, iterator_type> handles_;
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
  size_t workspace_bytes;
  decltype(AlgoPerfType().algo) algo;
  CudnnTensor x_tensor;
  CudnnFilterDescriptor filter_desc;
  CudnnTensor b_tensor;
  CudnnTensor y_tensor;
  CudnnConvolutionDescriptor conv_desc;

  lru_map<std::vector<int64_t>, AlgoPerfType> cached_benchmark_results { MAX_CACHED_ALGO_PERF_RESULTS };

  // note that conv objects are shared between execution frames, and a lock is needed to avoid multi-thread racing
  OrtMutex mutex;
};

enum : size_t {
  AlgoSearchWorkspaceSize = 32 * 1024 * 1024,
};

template <typename T>
class Conv : public CudaKernel, public ConvBase {
 public:
  Conv(const OpKernelInfo& info) : CudaKernel(info), ConvBase(info) {
    auto pads_size = pads_.size();
    ORT_ENFORCE(pads_size % 2 == 0);
    auto rank = pads_size / 2;
    for (size_t i = 0; i < rank; i++) {
      ORT_ENFORCE(pads_[i] == pads_[i + rank], "cudnn only supports symmetric padding");
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable CudnnConvState<cudnnConvolutionFwdAlgoPerf_t> s_;
};

}  // namespace cuda
}  // namespace onnxruntime
