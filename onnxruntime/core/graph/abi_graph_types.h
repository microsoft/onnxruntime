// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/onnx_protobuf.h"

// ORT C interface types for OrtGraphApi can't be in a namespace.
// We need to define them here so onnxruntime::Model can be created from OrtModel.

struct OrtValueInfo {
  std::string name;
  std::unique_ptr<OrtTypeInfo> type_info;
};

struct OrtOpAttr {
  ONNX_NAMESPACE::AttributeProto attr_proto;
};

struct OrtNode {
  enum class Type {
    kInvalid = 0,
    kEditorNode,
    kEpNode,
  };

  OrtNode() = default;
  explicit OrtNode(OrtNode::Type type) : type(type) {}
  virtual ~OrtNode() = default;
  OrtNode::Type type = OrtNode::Type::kInvalid;
};

struct OrtGraph {
  enum class Type {
    kInvalid = 0,
    kEditorGraph,
    kEpGraph,
  };

  OrtGraph() = default;
  explicit OrtGraph(OrtGraph::Type type) : type(type) {}
  virtual ~OrtGraph() = default;
  OrtGraph::Type type = OrtGraph::Type::kInvalid;
};
