// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include "core/graph/graph.h"
#include "core/graph/onnx_protobuf.h"
#include "core/training/graph_augmenter.h"

namespace onnxruntime {
namespace training {

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

  virtual bool CopyAttributes() const {
    return true;
  }

  // TODO: make this protected? Currently, compiler failure prevents it
  virtual GradientDef GetGradientDefs() = 0;

 protected:
  ArgDef I(const int i) {
    ORT_ENFORCE(i >= 0 && i < node_->InputDefs().size());
    return ArgDef(node_->InputDefs()[i]->Name(), node_->InputDefs()[i]->TypeAsProto());
  }

  ArgDef GI(const int i) {
    ORT_ENFORCE(i >= 0 && i < node_->InputDefs().size());
    return ArgDef(GradientName(node_->InputDefs()[i]->Name()), node_->InputDefs()[i]->TypeAsProto());
  }

  ArgDef GO(const int i) {
    ORT_ENFORCE(i >= 0 && i < node_->OutputDefs().size());
    return ArgDef(GradientName(node_->OutputDefs()[i]->Name()), node_->OutputDefs()[i]->TypeAsProto());
  }

  ArgDef IA(const std::string& argSuffix) {
    return ArgDef(Name(argSuffix), nullptr);
  }

  int GetSrcNodeOutputSize() {
    ORT_ENFORCE(node_ != nullptr);
    return (int)node_->OutputDefs().size();
  }

  // returns true if the input at index i of the node_ requires gradient
  bool IsGradientRequiredForSrcNodeInput(const int i) {
    ORT_ENFORCE(i >= 0 && i < node_->InputDefs().size());
    return gradient_outputs_.find(node_->InputDefs()[i]->Name()) != gradient_outputs_.end();
  }

  // returns true if the output at index i of the node_ has a gradient
  bool IsGradientAvailableForSrcNodeOutput(const int i) {
    ORT_ENFORCE(i >= 0 && i < node_->OutputDefs().size());
    return gradient_inputs_.find(node_->OutputDefs()[i]->Name()) != gradient_inputs_.end();
  }

  std::string Name(const std::string& name) {
    return unique_node_prefix_ + name;
  }

  const NodeAttributes& SrcNodeAttributes() {
    return node_->GetAttributes();
  }

 private:
  friend class GradientGraphBuilder;

  // Utility functions for gradient name computation. We don't expose them
  // in order to discourage the use of such names explicitly.
  static std::string GradientName(const std::string& name) {
    return name + "_grad";
  }

  std::string CreateUniqueNodePrefix() {
    ORT_ENFORCE(node_ != nullptr);
    auto name = node_->Name();
    std::stringstream unique_prefix;
    if (!name.empty()) {
      unique_prefix << name << "_";
    } else {
      unique_prefix << node_->OpType() << "_";
    }
    unique_prefix << node_->Index() << "_";
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
  virtual GradientDef GetGradientDefs() {
    return GradientDef();
  }
};

class UnSupportedGradientBuilder : public GradientBuilderBase {
  using GradientBuilderBase::GradientBuilderBase;
  virtual GradientDef GetGradientDefs() {
    ORT_ENFORCE(false, "Gradient should not be requested for this operator");
  }
};

}  // namespace training
}  // namespace onnxruntime
