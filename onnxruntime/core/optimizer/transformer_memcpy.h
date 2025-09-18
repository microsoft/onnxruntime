// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "gsl/gsl"

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/execution_provider.h"
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
  MemcpyTransformer(InlinedVector<gsl::not_null<const IExecutionProvider*>> providers,
                    const KernelRegistryManager& registry_manager)
      : GraphTransformer("MemcpyTransformer"),
        providers_(std::move(providers)),
        registry_manager_(std::cref(registry_manager)) {
  }

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  const InlinedVector<gsl::not_null<const IExecutionProvider*>> providers_;
  std::reference_wrapper<const KernelRegistryManager> registry_manager_;
};

}  // namespace onnxruntime
