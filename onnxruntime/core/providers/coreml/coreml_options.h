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

 public:
  explicit CoreMLOptions(uint32_t coreml_flags);

  CoreMLOptions(const ProviderOptions& options) {
    ValidateAndParseProviderOption(options);
  }
  bool RequireStaticShape() const { return require_static_shape_; }
  bool CreateMLProgram() const { return create_mlprogram_; }
  bool EnableOnSubgraph() const { return enable_on_subgraph_; }
  uint32_t ComputeUnits(uint32_t specific_flag = 0xffffffff) const { return compute_units_ & specific_flag; }

 private:
  void ValidateAndParseProviderOption(const ProviderOptions& options);
};
}  // namespace onnxruntime
