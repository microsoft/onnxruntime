// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include "core/graph/training/loss_func/bert_loss.h"

namespace onnxruntime {
namespace training {

TypeProto* BertLoss::GetLabelTypeProto(const NodeArg* prediction_arg,
                                       GraphAugmenter::GraphDefs& graph_defs) {
  ORT_ENFORCE(prediction_arg != nullptr, "GetLabelTypeProto's prediction_arg is nullptr");
  const auto* logits_type_proto = prediction_arg->TypeAsProto();
  const auto& dims = logits_type_proto->tensor_type().shape().dim();

  TypeProto* type_proto = graph_defs.CreateTypeProto();
  type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);

  auto* target_shape = type_proto->mutable_tensor_type()->mutable_shape();
  for (int i = 0; i < dims.size() - 1; ++i) {
    auto* target_dim = target_shape->add_dim();
    target_dim->CopyFrom(dims[i]);
  }

  return type_proto;
}

GraphAugmenter::GraphDefs BertLoss::operator()(const Graph& graph, const LossFunctionInfo& loss_func_info) {
  const std::string& loss_name = loss_func_info.loss_name;
  const VectorString& args = loss_func_info.loss_builder_args;
  ORT_ENFORCE(args.size() == 4, " Invalid loss_func_info for BertLoss.");
  const std::string& prediction_masked_lm = args[0];
  const std::string& prediction_next_sentence = args[1];
  const std::string& masked_lm_ids = args[2];
  const std::string& next_sentence_labels = args[3];

  std::vector<NodeDef> new_nodes;
  GraphAugmenter::GraphDefs graph_defs;

  // LabelSoftmaxCrossEntropy for marked_lm
  {
    const NodeArg* prediction_arg = graph.GetNodeArg(prediction_masked_lm);
    ORT_ENFORCE(prediction_arg != nullptr,
                "Masked_ML predition arg ", prediction_masked_lm, " is not found in the graph.");
    TypeProto* masked_lm_ids_type_proto = GetLabelTypeProto(prediction_arg, graph_defs);
    new_nodes.emplace_back(NodeDef("SparseSoftmaxCrossEntropy",  // Op
                                   {
                                       ArgDef(prediction_masked_lm),
                                       ArgDef(masked_lm_ids, masked_lm_ids_type_proto)},
                                   {
                                       ArgDef("loss_masked_lm")  // Outputs
                                   },
                                   NodeAttributes(),
                                   "masked_lm_loss"  // name
                                   ));
  }

  // LabelSoftmaxCrossEntropy for next_sentence
  {
    const NodeArg* ns_prediction_arg = graph.GetNodeArg(prediction_next_sentence);
    ORT_ENFORCE(ns_prediction_arg != nullptr,
                "Next sentence predition arg ", prediction_next_sentence, " is not found in the graph.");
    TypeProto* next_sentence_labels_type_proto = GetLabelTypeProto(ns_prediction_arg, graph_defs);

    new_nodes.emplace_back(NodeDef("SparseSoftmaxCrossEntropy",  // Op
                                   {
                                       ArgDef(prediction_next_sentence),
                                       ArgDef(next_sentence_labels,
                                              next_sentence_labels_type_proto)},
                                   {
                                       ArgDef("loss_next_sentence")  // Outputs
                                   },
                                   NodeAttributes(),
                                   "next_sentence_loss"  // name
                                   ));
  }

  // Add
  {
    new_nodes.emplace_back(NodeDef("Add",  // Op
                                   {
                                       ArgDef("loss_masked_lm"),
                                       ArgDef("loss_next_sentence")  // Inputs
                                   },
                                   {
                                       ArgDef(loss_name)  // Outputs
                                   },
                                   NodeAttributes(),
                                   "BertLoss_final_result"  // name
                                   ));
  }

  graph_defs.AddNodeDefs(new_nodes);
  graph_defs.AddGraphOutputs({loss_name});

  return graph_defs;
}
}  // namespace training
}  // namespace onnxruntime
