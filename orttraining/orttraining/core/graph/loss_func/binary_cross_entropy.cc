// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include "orttraining/core/graph/loss_func/binary_cross_entropy.h"

namespace onnxruntime {
namespace training {
GraphAugmenter::GraphDefs BinaryCrossEntropy::operator()(
    const Graph& graph,
    const LossFunctionInfo& loss_func_info) {
  const VectorString& args = loss_func_info.loss_builder_args;
  ORT_ENFORCE(args.size() == 2, " Invalid loss_func_info for BinaryCrossEntropy.");
  const std::string& prediction_name = args[0];
  const std::string& label_name = args[1];
  const std::string& loss_name = loss_func_info.loss_name;
  const std::string& prob_name = prediction_name + "_probability";

  GraphAugmenter::GraphDefs graph_defs;
  graph_defs.AddGraphInputs({label_name});
  graph_defs.AddGraphOutputs({loss_name});
  std::vector<NodeDef> new_nodes;
  {
    const NodeArg* prediction_arg = graph.GetNodeArg(prediction_name);
    ORT_ENFORCE(prediction_arg != nullptr, "Prediction arg ", prediction_name, " is not found in the graph.");
    TypeProto* label_type_proto = graph_defs.CopyTypeProto(prediction_arg);

    new_nodes.emplace_back(NodeDef(OpDef("BinaryCrossEntropy", kMSDomain, 1),  // Op
                                   {ArgDef(prediction_name),
                                    ArgDef(label_name, label_type_proto)},  // Inputs
                                   {ArgDef(loss_name, graph_defs.CreateTypeProto(std::array<const int64_t, 1>{1}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT)),
                                    ArgDef(prob_name)},  // Outputs
                                   NodeAttributes(),
                                   "BinaryCrossEntropy"  // name
                                   ));
  }
  graph_defs.AddNodeDefs(new_nodes);
  return graph_defs;
}

}  // namespace training
}  // namespace onnxruntime
