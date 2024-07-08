// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/vulkan/vulkan_kernel.h"

namespace onnxruntime {
namespace vulkan {
class Sigmoid final : public VulkanKernel {
 public:
  explicit Sigmoid(const OpKernelInfo& info);
  ~Sigmoid();

  Status Compute(OpKernelContext* context) const override;

  static bool IsSupported(const onnxruntime::Node& /*node*/) {
    // implement any non-data type checks here
    return true;
  }

 private:
  void SetupLayer() const;

  const int32_t data_type_;
  const int32_t ncnn_index_{-1};
  // if input has fixed size we can re-use the pipeline in Compute
  std::optional<LayerPipeline> fixed_size_pipeline_;

  // Layer has to be mutable to allow creating the pipeline based on input shapes when fixed_size_input_ is false.
  mutable ncnn::Layer* ncnn_layer_{nullptr};
};
}  // namespace vulkan
}  // namespace onnxruntime
