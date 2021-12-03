// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/framework/kernel_registry_manager.h"

namespace onnxruntime {

/**
@Class AnnotateNodeArg
# FIXME: add doc
*/
class AnnotateNodeArg : public GraphTransformer {
 public:
  explicit AnnotateNodeArg(const KernelRegistryManager& registry_manager) noexcept;

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  std::reference_wrapper<const KernelRegistryManager> registry_manager_;
};

}  // namespace onnxruntime
