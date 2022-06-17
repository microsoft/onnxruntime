// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <utility>

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
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
  explicit NhwcTransformer(const InlinedHashSet<std::string_view>& compatible_execution_providers,
                           AllocatorPtr cpu_allocator) noexcept
      : GraphTransformer("NhwcTransformer", compatible_execution_providers), cpu_allocator_(std::move(cpu_allocator)) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
