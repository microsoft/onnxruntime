// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

class CoreMLOptions {
 private:
  bool require_static_shape_{false};
  bool create_mlprogram_{false};
  bool enable_on_subgraph_{false};
  uint32_t compute_units_{0};
  std::string strategy_;
  bool profile_compute_plan_{false};
  bool allow_low_precision_accumulation_on_gpu_{false};

 public:
  explicit CoreMLOptions(uint32_t coreml_flags);

  CoreMLOptions(const ProviderOptions& options) {
    ValidateAndParseProviderOption(options);
  }
  bool RequireStaticShape() const { return require_static_shape_; }
  bool CreateMLProgram() const { return create_mlprogram_; }
  bool EnableOnSubgraph() const { return enable_on_subgraph_; }
  uint32_t ComputeUnits(uint32_t specific_flag = 0xffffffff) const { return compute_units_ & specific_flag; }
  bool AllowLowPrecisionAccumulationOnGPU() const { return allow_low_precision_accumulation_on_gpu_; }
  bool UseStrategy(std::string_view strategy) const { return strategy_ == strategy; }
  bool ProfileComputePlan() const { return profile_compute_plan_ && create_mlprogram_; }

 private:
  void ValidateAndParseProviderOption(const ProviderOptions& options);
};
}  // namespace onnxruntime
