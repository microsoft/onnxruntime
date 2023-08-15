// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef PRINT_TOLERANCE_ERRORS

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class PrintToleranceErrorsTransformer : public GraphTransformer {
 public:
  PrintToleranceErrorsTransformer(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("PrintToleranceErrorsTransformer", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime

#endif
