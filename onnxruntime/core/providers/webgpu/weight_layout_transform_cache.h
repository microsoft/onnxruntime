// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <mutex>
#include <string>
#include "core/framework/tensor.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace webgpu {

// Cache manager for transformed weights
// Owned by WebGpuExecutionProvider
class WeightLayoutTransformCache {
 public:
  WeightLayoutTransformCache() = default;
  ~WeightLayoutTransformCache() = default;

  // Get transformed weight from cache (nullptr if not found)
  const Tensor* GetTransformedWeight(const std::string& weight_name,
                                     const std::string& format_descriptor) const;

  // Add transformed weight to cache
  void AddTransformedWeight(const std::string& weight_name,
                            const std::string& format_descriptor,
                            Tensor&& tensor);

  // Clear cache (must be called before BufferManager is destroyed)
  void Clear();

 private:
  std::string MakeCacheKey(const std::string& weight_name,
                           const std::string& format) const {
    return weight_name + ":" + format;
  }

  mutable std::mutex mutex_;
  std::unordered_map<std::string, Tensor> cache_;
};

}  // namespace webgpu
}  // namespace onnxruntime
