// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_config.h"
namespace onnxruntime {

/**
@Class PropagateCastOps

Propagate FP16 Cast operations up the graph and FP32 Cast operations down the graph

*/
class PropagateCastOps : public GraphTransformer {
 public:
  /*
   * The collection FP16AllowOps, specifies for a given propagate_cast_ops level, a collection of node op_types that
   * the code is allowed to propage Cast operations across. The user may specify a custom list of optypes using level 0.
   * The opcodes are split into multiple levels. Cast propagation is done based on the level. Level 2 op code
   * list includes Level 1 list also.
   */
  typedef std::vector<std::unordered_set<std::string>> FP16AllowOps;

  PropagateCastOps(GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy strategy =
                       GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::FloodFill,
                   size_t level = 0, const std::vector<std::string>& allow_list = {},
                   const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept;

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  size_t level_;
  FP16AllowOps fp16_allow_ops_;
  GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy strategy_;
};

}  // namespace onnxruntime
