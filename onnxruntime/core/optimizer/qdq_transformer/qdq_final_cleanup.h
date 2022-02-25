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
    using fp32.

    e.g. if we have Op -> Q -> DQ -> Op2 and no more QDQ processing to run, the Q -> DQ can potentially be removed.

    The impact on performance and accuracy of removing the pair will depend on the model.

    If it was quantized with Quantization Aware Training (QAT) it may be better to pay the performance cost of keeping
    the pair as the model was trained with a round-trip from float -> 8-bit -> float.

    If the model was quantized with Post Training Quantization (PTQ) it is most likely better to remove the pair as the
    loss of precision from the round trip of float -> 8-bit -> float was not present when the model was trained.

    As we have no knowledge of how the model was quantized we require the user to specify an option to enable this
    transformer.
    */
class QDQFinalCleanupTransformer : public GraphTransformer {
 public:
  QDQFinalCleanupTransformer(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("QDQFinalCleanupTransformer", compatible_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
