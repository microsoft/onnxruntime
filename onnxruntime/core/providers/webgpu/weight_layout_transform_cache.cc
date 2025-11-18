// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/weight_layout_transform_cache.h"

namespace onnxruntime {
namespace webgpu {

const Tensor* WeightLayoutTransformCache::GetTransformedWeight(
    const std::string& weight_name,
    const std::string& format_descriptor) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string cache_key = MakeCacheKey(weight_name, format_descriptor);
  auto it = cache_.find(cache_key);
  if (it != cache_.end()) {
    return &it->second;
  }
  return nullptr;
}

void WeightLayoutTransformCache::AddTransformedWeight(
    const std::string& weight_name,
    const std::string& format_descriptor,
    Tensor&& tensor) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string cache_key = MakeCacheKey(weight_name, format_descriptor);
  cache_[cache_key] = std::move(tensor);
}

void WeightLayoutTransformCache::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.clear();
}

}  // namespace webgpu
}  // namespace onnxruntime
