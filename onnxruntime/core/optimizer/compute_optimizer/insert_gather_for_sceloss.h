// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE
#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/compute_optimizer/common.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime::optimizer::compute_optimizer {

/**
 * @brief

 */
class InsertGatherBeforeSceLoss : public GraphTransformer {
 public:
  InsertGatherBeforeSceLoss(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("InsertGatherBeforeSceLoss", compatible_execution_providers) {
  }

  /**
   * @brief
   */
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime::optimizer::compute_optimizer
#endif
