// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include "core/framework/data_types.h"
#include "orttraining/core/graph/graph_augmenter.h"

namespace onnxruntime {
namespace training {

struct LossFunctionInfo {
  LossFunctionInfo() {}
  LossFunctionInfo(const OpDef& op_def, const std::string& loss_name, const VectorString& loss_builder_args)
      : op_def(op_def), loss_name(loss_name), loss_builder_args(loss_builder_args) {}

  OpDef op_def;
  std::string loss_name;
  VectorString loss_builder_args;
};

struct ILossFunction {
  virtual GraphAugmenter::GraphDefs operator()(const Graph& graph, const LossFunctionInfo& loss_func_info) = 0;
  virtual ~ILossFunction(){};
};

TypeProto* GetSparseTypeProto(const NodeArg* input_arg,
                              ONNX_NAMESPACE::TensorProto_DataType data_type,
                              GraphAugmenter::GraphDefs& graph_defs);

}  // namespace training
}  // namespace onnxruntime
