// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/optimizer/graph_transformer.h"
#include "core/common/logging/logging.h"

using namespace onnxruntime::logging;

namespace onnxruntime {

/**
@Class MemcpyTransformer

Transformer that inserts nodes to copy memory between devices when needed.
*/
class MemcpyTransformer : public GraphTransformer {
 public:
  MemcpyTransformer(const std::vector<std::string>& provider_types, const KernelRegistryManager& registry_manager, const Logger& logger)
      : GraphTransformer("MemcpyTransformer"), provider_types_{provider_types}, registry_manager_{registry_manager}, logger_{logger} {}

 private:
  common::Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;

  const std::vector<std::string> provider_types_;
  const KernelRegistryManager& registry_manager_;
  const Logger& logger_;
};

}  // namespace onnxruntime
