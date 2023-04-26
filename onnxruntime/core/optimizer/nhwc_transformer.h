// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class NhwcTransformer

Transformer that optimizes the graph by using NHWC nodes instead of NCHW nodes
and inserts nodes to transpose tensors as needed.
*/
class NhwcTransformer : public GraphTransformer {
 private:
  AllocatorPtr cpu_allocator_;

 public:
  explicit NhwcTransformer(AllocatorPtr cpu_allocator, std::shared_ptr<KernelRegistry> cpu_kernel_registry) noexcept;

  /**
   * @brief Usually called right after constructor, it shows whether
   *        this transformer should be used under current hardware configuration.
   * 
   * @return whether this transformer would be useful under current hardware config 
   */
  bool IsActive();

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
