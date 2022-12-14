// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CUDA

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * @brief Fuse sub-graphs containing simple Ops (such as elementwise Ops) into TorchScript Ops for nvFuser fallback
 * in CUDA EP.
 *
 */
class TorchScriptFusion : public GraphTransformer {
 public:
  TorchScriptFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("TorchScriptFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  bool IsSupportedNode(const Graph& graph, const Node& node) const;
};

}  // namespace onnxruntime

#endif  // USE_CUDA
