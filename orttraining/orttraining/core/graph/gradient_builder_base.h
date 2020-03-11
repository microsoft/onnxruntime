// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include "core/graph/graph.h"
#include "core/graph/onnx_protobuf.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "onnx/defs/attr_proto_util.h"

namespace onnxruntime {
namespace training {

using Dimension = onnx::TensorShapeProto_Dimension;

void ComputeBroadcastBackwardAxes(
    const std::vector<Dimension>& A_dims,
    const std::vector<Dimension>& B_dims,
    std::vector<int64_t>* A_axes,
    std::vector<int64_t>* B_axes);

std::vector<Dimension> GetShape(const ArgDef& arg_def);

typedef std::vector<NodeDef> GradientDef;

class GradientBuilderBase {
 public:
  GradientBuilderBase(
      const Node* node,
      const std::unordered_set<std::string>& gradient_inputs,
      const std::unordered_set<std::string>& gradient_outputs)
      : node_(node), gradient_inputs_(gradient_inputs), gradient_outputs_(gradient_outputs) {
    unique_node_prefix_ = CreateUniqueNodePrefix();
  }

  virtual ~GradientBuilderBase() {}

  GradientDef GetGradientDefs() const {
    GradientDef node_defs = GetGradientDefsImpl();
    for (size_t i = 0; i < node_defs.size(); ++i) {
      NodeDef& node_def = node_defs[i];
      if (node_def.name.empty()) {
        node_def.name = Name(node_def.op_type + "_" + std::to_string(i));
      }
    }
    return node_defs;
  }

  static std::string GradientName(const std::string& name) {
    return name + "_grad";
  }

 protected:
  virtual GradientDef GetGradientDefsImpl() const = 0;

  ArgDef I(const size_t i) const {
    ORT_ENFORCE(i < node_->InputDefs().size());
    return ArgDef(node_->InputDefs()[i]->Name(), node_->InputDefs()[i]->TypeAsProto());
  }

  ArgDef O(const size_t i) const {
    ORT_ENFORCE(i < node_->OutputDefs().size());
    return ArgDef(node_->OutputDefs()[i]->Name(), node_->OutputDefs()[i]->TypeAsProto());
  }

  ArgDef GI(const size_t i) const {
    ORT_ENFORCE(i < node_->InputDefs().size());
    return ArgDef(GradientName(node_->InputDefs()[i]->Name()), node_->InputDefs()[i]->TypeAsProto());
  }

  ArgDef GO(const size_t i) const {
    ORT_ENFORCE(i < node_->OutputDefs().size());
    return ArgDef(GradientName(node_->OutputDefs()[i]->Name()), node_->OutputDefs()[i]->TypeAsProto());
  }

  ArgDef IA(const std::string& argSuffix, const TypeProto* type_proto = nullptr) const {
    return ArgDef(Name(argSuffix), type_proto);
  }

  const TypeProto* IType(const size_t i) const {
    ORT_ENFORCE(i < node_->InputDefs().size());
    return node_->InputDefs()[i]->TypeAsProto();
  }

  const TypeProto* OType(const size_t i) const {
    ORT_ENFORCE(i < node_->OutputDefs().size());
    return node_->OutputDefs()[i]->TypeAsProto();
  }

  int GetSrcNodeInputSize() const {
    ORT_ENFORCE(node_ != nullptr);
    return (int)node_->InputDefs().size();
  }

  int GetSrcNodeOutputSize() const {
    ORT_ENFORCE(node_ != nullptr);
    return (int)node_->OutputDefs().size();
  }

  // returns true if the input at index i of the node_ requires gradient
  bool IsGradientRequiredForSrcNodeInput(const size_t i) const {
    return i < node_->InputDefs().size() &&
           gradient_outputs_.find(node_->InputDefs()[i]->Name()) != gradient_outputs_.end();
  }

  // returns true if the output at index i of the node_ has a gradient
  bool IsGradientAvailableForSrcNodeOutput(const size_t i) const {
    return i < node_->OutputDefs().size() &&
           gradient_inputs_.find(node_->OutputDefs()[i]->Name()) != gradient_inputs_.end();
  }

  std::string Name(const std::string& name) const {
    return unique_node_prefix_ + name;
  }

  const NodeAttributes& SrcNodeAttributes() const {
    return node_->GetAttributes();
  }

  const std::string& SrcNodeOpType() const {
    return node_->OpType();
  }

  static NodeDef ConstantValueNode(const std::vector<int64_t>& values, const std::string& arg_name) {
    ONNX_NAMESPACE::TensorProto t_proto;
    t_proto.add_dims(values.size());
    t_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    for (auto value : values) {
      t_proto.add_int64_data(value);
    }

    return NodeDef("Constant",
                   {},
                   {ArgDef(arg_name, nullptr)},
                   {ONNX_NAMESPACE::MakeAttribute("value", t_proto)});
  }

  static NodeDef ConstantValueNode(float value, const std::string& arg_name) {
    ONNX_NAMESPACE::TensorProto t_proto;
    t_proto.add_dims(1);
    t_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    t_proto.add_float_data(value);

    return NodeDef("Constant",
                   {},
                   {ArgDef(arg_name, nullptr)},
                   {ONNX_NAMESPACE::MakeAttribute("value", t_proto)});
  }

  static NodeDef ZeroConstantNode() {
    return ConstantValueNode(0.0f, "ZeroConstant");
  }

  static NodeDef OneConstantNode() {
    return ConstantValueNode(1.0f, "OneConstant");
  }

  void HandleBroadcasting(const ArgDef& input_grad,
                          const ArgDef& target,
                          const ArgDef& output_grad,
                          const std::vector<int64_t>& reduce_axes,
                          std::vector<NodeDef>& output) const;

 private:
  friend class GradientGraphBuilder;

  std::string CreateUniqueNodePrefix() {
    ORT_ENFORCE(node_ != nullptr);
    auto name = node_->Name();
    std::stringstream unique_prefix;

    if (!name.empty()) {
      unique_prefix << name << "_Grad/";
    } else {
      unique_prefix << node_->OpType() << "_" << node_->Index() << "_Grad/";
    }
    return unique_prefix.str();
  }

  const Node* node_;
  std::string unique_node_prefix_;

  // contains set of output arg names of node_ which is provided as gradient input to the bw node
  std::unordered_set<std::string> gradient_inputs_;

  // contains set of input arg names of node_ which requires gradient
  std::unordered_set<std::string> gradient_outputs_;
};

class EmptyGradientBuilder : public GradientBuilderBase {
  using GradientBuilderBase::GradientBuilderBase;
  GradientDef GetGradientDefsImpl() const override {
    return GradientDef();
  }
};

class UnSupportedGradientBuilder : public GradientBuilderBase {
  using GradientBuilderBase::GradientBuilderBase;
  GradientDef GetGradientDefsImpl() const override {
    ORT_ENFORCE(false, "Gradient should not be requested for this operator");
  }
};

}  // namespace training
}  // namespace onnxruntime
