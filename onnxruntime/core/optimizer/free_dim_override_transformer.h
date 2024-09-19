// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

struct FreeDimensionOverride;

/**
@Class FreeDimensionOverrideTransformer

Transformer that overrides free dimensions in the graph with the specific value
that matches the denotation for that dimension.
*/
class FreeDimensionOverrideTransformer : public GraphTransformer {
 public:
  explicit FreeDimensionOverrideTransformer(gsl::span<const FreeDimensionOverride> overrides_to_apply);

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  std::map<std::string, int64_t> dimension_override_by_denotation_;
  std::map<std::string, int64_t> dimension_override_by_name_;
};

}  // namespace onnxruntime
