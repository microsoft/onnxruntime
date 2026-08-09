// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRITON

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

struct TritonFusionConfig {
  struct OpInfo {
    std::string domain = "";
    std::vector<ONNX_NAMESPACE::OperatorSetVersion> versions;
    bool is_no_op = false;
    std::unordered_map<std::string, std::string> conditions = {};
    bool ignore_min_nodes = false;
  };

  TritonFusionConfig(std::string_view config_json = "{}");

  bool IsSupported(const Graph& graph, const Node& node) const;
  bool IsNoOp(const Node& node) const;
  bool IgnoreMinNodes(const Node& node) const;
  const ONNX_NAMESPACE::TensorProto* TryGetInitializer(const Graph& graph, const Node& node, NodeArg* node_arg) const;

  std::unordered_map<std::string, OpInfo> ops;
  std::string initializer = "none";
  size_t min_nodes = 2;
};

class TritonFusion : public GraphTransformer {
 public:
  TritonFusion(std::string_view config_json = "{}",
               const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("TritonFusion", compatible_execution_providers), config_(config_json) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  bool IsSupportedNode(const Graph& graph, const Node& node) const;

  TritonFusionConfig config_;
};

}  // namespace onnxruntime

#endif  // ENABLE_TRITON
