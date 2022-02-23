// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
    @Class QDQFinalCleanupTransformer

    Remove any remaining back-to-back QuantizeLinear and DequantizeLinear pairs.

    This is the final cleanup where no quantized operator was available and we're going to run the operators
    using fp32. As such there's no point going between quantized and fp32.

    e.g. if we have Op -> Q -> DQ -> Op2 and no more QDQ processing to run, the Q -> DQ is pointless
    (assuming you don't want to lose accuracy and performance to run them).

    TODO: I'm sure there's a scenario where we may have a DQ -> Q remaining. Check when that happens to document
    and add support for that pair
    */
class QDQFinalCleanupTransformer : public GraphTransformer {
 public:
  QDQFinalCleanupTransformer(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("QDQFinalCleanupTransformer", compatible_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
