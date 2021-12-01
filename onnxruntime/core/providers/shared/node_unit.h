// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
// #include "core/graph/basic_types.h"
// Need move Node::NodeConstIterator out of Node for forward declaration
#include "core/graph/graph.h"

namespace onnxruntime {

// template <typename Container>
// class ConstPointerContainer;
// class Node;
// class NodeArg;
// class Path;
class GraphViewer;

namespace QDQ {
struct NodeGroup;
}

class INodeUnit {
 public:
  enum class Type : uint8_t {
    Node,
    QDQ
  };

  virtual ~INodeUnit() = default;

  virtual const ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept = 0;
  virtual const ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept = 0;

  virtual const std::string& OpType() const noexcept = 0;
  virtual int SinceVersion() const noexcept = 0;
  virtual const std::string& Domain() const noexcept = 0;
  virtual const Path& ModelPath() const noexcept = 0;
  virtual const std::string& Name() const noexcept = 0;

  virtual const Node& GetNode() const noexcept = 0;

  // virtual size_t GetInputEdgesCount() const noexcept = 0;
  virtual NodeIndex Index() const noexcept = 0;

  virtual ProviderType GetExecutionProviderType() const noexcept = 0;

  // virtual Node::NodeConstIterator OutputNodesBegin() const noexcept = 0;
  // virtual Node::NodeConstIterator OutputNodesEnd() const noexcept = 0;

  virtual const std::vector<const Node*> GetAllNodes() const noexcept = 0;

  virtual Type UnitType() const noexcept = 0;
};

const std::unique_ptr<INodeUnit> CreateNodeUnit(const Node& node);
const std::unique_ptr<INodeUnit> CreateQDQNodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group);
}  // namespace onnxruntime
