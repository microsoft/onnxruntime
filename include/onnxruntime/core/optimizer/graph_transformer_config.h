// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

struct GraphTransformerConfiguration {
  struct PropagateCastOpsConfiguration {
    // Propagate FP16 Cast operations up and FP32 operations down
    int level{-1};
    // Cast propagation strategy. One strategy is to insert casts around all the nodes with the allowed opcodes
    // and reduce, remove redundent and back-to-back casts, and
    // the other is to propagate casts using flood-fill approach, expanding float16 regions in the graph
    // traversinvg the graph up/down.
    enum class Strategy {
      InsertAndReduce = 0,
      FloodFill = 1
    };
    Strategy strategy = Strategy::InsertAndReduce;
    // List of allowed opcodes to consider as safe to execute in float16, while moving cast operations
    std::vector<std::string> allow;
    // Whether allow fusion of layer norm subgraph if doing so will cause modified precision.
  };  // Whether to enable GELU approximation which is faster but produces different results.
  PropagateCastOpsConfiguration propagate_cast_ops_config;
};

}  // namespace onnxruntime