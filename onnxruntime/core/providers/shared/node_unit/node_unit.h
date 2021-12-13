// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <optional>

#include "core/graph/basic_types.h"

namespace onnxruntime {

class Graph;
class GraphViewer;
class Node;
class NodeArg;
class Path;

namespace QDQ {
struct NodeGroup;
}

class NodeUnit {
 public:
  enum class Type : uint8_t {
    Node,
    QDQ
  };

  struct IODef {
    struct QuantParam {
      const NodeArg* scale{nullptr};
      const NodeArg* zero_point{nullptr};
    };

    const NodeArg& node_arg;
    const std::optional<QuantParam> quant_param;
  };

 public:
  explicit NodeUnit(const Node& node);
  ~NodeUnit();

  Type UnitType() const noexcept { return type_; }

  const std::vector<IODef>& InputDefs() const noexcept { return input_defs_; }
  const std::vector<IODef>& OutputDefs() const noexcept { return output_defs_; }

  const std::string& Domain() const noexcept;
  const std::string& OpType() const noexcept;
  const std::string& Name() const noexcept;
  int SinceVersion() const noexcept;
  NodeIndex Index() const noexcept;
  const Path& ModelPath() const noexcept;
  ProviderType GetExecutionProviderType() const noexcept;

  const Node& GetNode() const noexcept { return node_; }

  const std::vector<const Node*> GetAllNodes() const noexcept { return nodes_; }

 private:
  std::vector<IODef> input_defs_;
  std::vector<IODef> output_defs_;

  const std::vector<const Node*> nodes_;  // all nodes in this NodeUnit
  const Node& node_;                      // target Node
  Type type_;

  void InitForNode();  // Initializing for single Node
};

}  // namespace onnxruntime
