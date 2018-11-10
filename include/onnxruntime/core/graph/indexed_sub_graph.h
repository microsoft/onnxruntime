// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/graph/basic_types.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {

class OpKernel;
class OpKernelInfo;
// Sub-graph data structure.
// It contains a node index array covered by <*this> sub-graph,
// and contains meta definition needed for customizing <*this>
// sub-graph as a FunctionProto, which could be serialized/saved
// to a model file.
struct IndexedSubGraph {
  struct MetaDef {
    // Name of customized Sub-Graph/FunctionProto
    std::string name;
    // Domain of customized Sub-Graph/FunctionProto
    std::string domain;
    // Since version of customized Sub-Graph/FunctionProto.
    int since_version;
    // Status of customized Sub-Graph/FunctionProto.
    ONNX_NAMESPACE::OperatorStatus status;
    // Inputs of customized Sub-Graph/FunctionProto.
    std::vector<std::string> inputs;
    // Outputs of customized Sub-Graph/FunctionProto.
    std::vector<std::string> outputs;
    // Attributes of customized Sub-Graph/FunctionProto.
    std::vector<std::string> attributes;
    // Doc string of customized Sub-Graph/FunctionProto.
    std::string doc_string;
  };

  // Nodes covered by <*this> sub-graph.
  // The indexes are from parent graph.
  std::vector<onnxruntime::NodeIndex> nodes;

  // Meta definition needed for customizing <*this>
  // sub-graph as a FunctionProto, which could be serialized/saved
  // to a model file. It's needed IF AND ONLY IF there're multiple
  // indexes contained in <nodes> above.

  void SetMetaDef(std::unique_ptr<MetaDef>& meta_def_) {
    meta_def = std::move(meta_def_);
  }

  const MetaDef* GetMetaDef() const {
    return meta_def.get();
  }

 private:
  // Sub-graph meta definition.
  std::unique_ptr<MetaDef> meta_def;
};

}  // namespace onnxruntime
