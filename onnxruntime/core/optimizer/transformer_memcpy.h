// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class MemcpyTransformer

Transformer that inserts nodes to copy memory between devices when needed.
*/
class MemcpyTransformer : public GraphTransformer {
 public:
  MemcpyTransformer(const std::vector<std::string>& provider_types, const KernelRegistryManager& registry_manager)
      : GraphTransformer("MemcpyTransformer"), provider_types_(provider_types), registry_manager_(std::cref(registry_manager)) {}

 private:
  common::Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  const std::vector<std::string> provider_types_;
  std::reference_wrapper<const KernelRegistryManager> registry_manager_;
};

}  // namespace onnxruntime
