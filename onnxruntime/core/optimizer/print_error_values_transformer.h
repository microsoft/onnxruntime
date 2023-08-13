// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef PRINT_ERROR_VALUES

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class PrintErrorValuesTransformer : public GraphTransformer {
 public:
  PrintErrorValuesTransformer(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("PrintErrorValuesTransformer", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime

#endif
