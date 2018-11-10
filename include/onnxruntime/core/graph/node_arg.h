// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {

// Node argument definition, for both input and output,
// including arg name, arg type (contains both type and shape).
//
// Design Question: in my opinion, shape should not be part of type.
// We may align the protobuf design with our operator registry interface,
// which has type specified for each operator, but no shape. Well, shape
// should be inferred with a separate shape inference function given
// input shapes, or input tensor data sometimes.
// With shape as part of type (current protobuf design),
// 1) we'll have to split the "TypeProto" into type and shape in this internal
// representation interface so that it could be easily used when doing type
// inference and matching with operator registry.
// 2) SetType should be always called before SetShape, otherwise, SetShape()
// will fail. Because shape is located in a TypeProto.
// Thoughts?
//
class NodeArg {
 public:
  // Constructor by specifying node arg name and type&shape which is
  // optional. This is called when loading a <Graph> from <GraphProto>
  // normally.
  NodeArg(const std::string& name,
          const ONNX_NAMESPACE::TypeProto* p_arg_type);

  NodeArg(NodeArg&& other) = default;

  // Get node arg name.
  const std::string& Name() const noexcept;

  // Get node arg type.
  ONNX_NAMESPACE::DataType Type() const noexcept;
  const ONNX_NAMESPACE::TypeProto* TypeAsProto() const noexcept;

  // Get node arg shape.
  // Return null pointer if there's no shape specified.
  const ONNX_NAMESPACE::TensorShapeProto* Shape() const;

  // Set node arg shape.
  // Shape could only be set after setting type since shape information
  // now is part of TypeProto.
  void SetShape(const ONNX_NAMESPACE::TensorShapeProto& shape);

  // validate and merge type [and shape] info from input_type.
  // if there is existing type info that can't be cleanly updated return an error.
  common::Status UpdateTypeAndShape(const ONNX_NAMESPACE::TypeProto& input_type);

  // validate and merge type [and shape] info from input_type.
  // if there is existing type info that can't be cleanly updated return an error.
  common::Status UpdateTypeAndShape(const NodeArg& node_arg);

  // Get node arg info proto.
  const NodeArgInfo& ToProto() const noexcept { return node_arg_info_; }

  // Indicates whether <*this> node arg exists or not.
  // Optional inputs are allowed in ONNX. Empty arg name represents
  // a non-existing input argument.
  bool Exists() const noexcept;

 private:
  ONNXRUNTIME_DISALLOW_COPY_AND_ASSIGNMENT(NodeArg);
  friend class Graph;

  void SetType(ONNX_NAMESPACE::DataType p_type);
  void SetType(const ONNX_NAMESPACE::TypeProto& type_proto);

  NodeArg& operator=(NodeArg&& other) = delete;

  // Node arg PType.
  ONNX_NAMESPACE::DataType type_;

  // Node arg name, type and shape.
  NodeArgInfo node_arg_info_;

  // Flag indicates whether <*this> node arg exists or not.
  bool exists_;
};
}  // namespace onnxruntime
