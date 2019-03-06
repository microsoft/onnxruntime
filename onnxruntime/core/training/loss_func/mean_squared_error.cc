// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/training/loss_func/mean_squared_error.h"
#include <unordered_map>

namespace onnxruntime {
namespace training {

using namespace std;
using namespace onnxruntime::common;

GraphAugmenter::GraphDefs MeanSquaredError(const LossFunctionInfo& loss_func_info) {
  GraphAugmenter::GraphDefs graph_defs;

  graph_defs.AddGraphOutputs({loss_func_info.loss_name_});

  vector<NodeDef> new_nodes;
  // Sub
  {
    new_nodes.emplace_back(NodeDef("Sub",                    // Op
                                   "MeanSquaredError_diff",  // name
                                   {
                                       ArgDef(loss_func_info.prediction_name_),
                                       ArgDef(loss_func_info.label_name_)},  // Inputs
                                   {
                                       ArgDef("MeanSquaredError_diff")  // Outputs
                                   }));
  }
  // Pow
  {
    onnx::TensorProto tensor_proto;
    tensor_proto.add_dims(1);
    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_proto.add_float_data(2.f);
    tensor_proto.set_name("MeanSquaredError_exponent");
    graph_defs.AddInitializers({tensor_proto});

    new_nodes.emplace_back(NodeDef("Pow",                   // Op
                                   "MeanSquaredError_pow",  // name
                                   {
                                       ArgDef("MeanSquaredError_diff"),  // Inputs
                                       ArgDef("MeanSquaredError_exponent")},
                                   {
                                       ArgDef("MeanSquaredError_diff_square")  // Outputs
                                   }));
  }
  // ReduceMean
  {
    NodeAttributes attributes;
    {
      onnx::AttributeProto att;
      att.set_name("axes");
      att.set_type(onnx::AttributeProto::INTS);
      att.add_ints(1);
      attributes["axes"] = att;
    }
    {
      onnx::AttributeProto att;
      att.set_name("keepdims");
      att.set_type(onnx::AttributeProto::INT);
      att.set_i(0);
      attributes["keepdims"] = att;
    }
    new_nodes.emplace_back(NodeDef("ReduceMean",                    // Op
                                   "MeanSquaredError_reduce_mean",  // name
                                   {
                                       ArgDef("MeanSquaredError_diff_square")},  // Inputs
                                   {
                                       ArgDef(loss_func_info.loss_name_)  // Outputs
                                   },
                                   attributes));
  }

  graph_defs.AddNodeDefs(new_nodes);
  return graph_defs;
}
}  // namespace training
}  // namespace onnxruntime
