// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include "orttraining/core/graph/loss_func/bert_loss.h"
#include "onnx/defs/attr_proto_util.h"

namespace onnxruntime {
namespace training {

TypeProto* BertLoss::GetMaskedLMTypeProto(const NodeArg* prediction_arg,
                                          ONNX_NAMESPACE::TensorProto_DataType data_type,
                                          GraphAugmenter::GraphDefs& graph_defs) {
  ORT_ENFORCE(prediction_arg != nullptr, "GetMaskedPredictionTypeProto's prediction_arg is nullptr");
  const auto* logits_type_proto = prediction_arg->TypeAsProto();
  const auto& dims = logits_type_proto->tensor_type().shape().dim();

  TypeProto* type_proto = graph_defs.CreateTypeProto();
  type_proto->mutable_tensor_type()->set_elem_type(data_type);

  auto* target_shape = type_proto->mutable_tensor_type()->mutable_shape();
  // Batch size.
  target_shape->add_dim()->CopyFrom(dims[0]);
  // Prediction count.
  target_shape->add_dim()->set_dim_param("dynamic_prediction_count");

  return type_proto;
}

TypeProto* BertLoss::GetGatheredPredictionTypeProto(const NodeArg* prediction_arg,
                                                    GraphAugmenter::GraphDefs& graph_defs) {
  ORT_ENFORCE(prediction_arg != nullptr, "GetMaskedPredictionTypeProto's prediction_arg is nullptr");
  const auto* logits_type_proto = prediction_arg->TypeAsProto();
  const auto& dims = logits_type_proto->tensor_type().shape().dim();

  TypeProto* type_proto = graph_defs.CreateTypeProto();
  type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto* target_shape = type_proto->mutable_tensor_type()->mutable_shape();
  // Batch size.
  target_shape->add_dim()->CopyFrom(dims[0]);
  // Prediction count.
  target_shape->add_dim()->set_dim_param("dynamic_prediction_count");
  // Vocab size.
  target_shape->add_dim()->CopyFrom(dims[2]);

  return type_proto;
}

TypeProto* BertLoss::GetLossTypeProto(GraphAugmenter::GraphDefs& graph_defs) {
  return graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
}

GraphAugmenter::GraphDefs BertLoss::operator()(const Graph& graph, const LossFunctionInfo& loss_func_info) {
  const std::string& total_loss = loss_func_info.loss_name;
  const VectorString& args = loss_func_info.loss_builder_args;
  ORT_ENFORCE(args.size() == 7, " Invalid loss_func_info for BertLoss.");
  const std::string& prediction_masked_lm = args[0];
  const std::string& prediction_next_sentence = args[1];
  const std::string& masked_lm_positions = args[2];
  const std::string& masked_lm_ids = args[3];
  const std::string& next_sentence_labels = args[4];
  const std::string& mlm_loss = args[5];
  const std::string& nsp_loss = args[6];

  std::vector<NodeDef> new_nodes;
  GraphAugmenter::GraphDefs graph_defs;
  // LabelSoftmaxCrossEntropy for masked_lm
  {
    const NodeArg* prediction_arg = graph.GetNodeArg(prediction_masked_lm);
    ORT_ENFORCE(prediction_arg != nullptr,
                "Masked_ML prediction arg ", prediction_masked_lm, " is not found in the graph.");
    TypeProto* masked_lm_int64_type_proto = GetMaskedLMTypeProto(prediction_arg,
                                                                 ONNX_NAMESPACE::TensorProto_DataType_INT64,
                                                                 graph_defs);

    new_nodes.emplace_back(NodeDef("Unsqueeze",
                                   {ArgDef(masked_lm_positions, masked_lm_int64_type_proto)},
                                   {ArgDef("masked_lm_positions_unsqueezed")},
                                   {ONNX_NAMESPACE::MakeAttribute("axes", std::vector<int64_t>{static_cast<int64_t>(2)})},
                                   "Mask_LM_Positions_Unsqueezed"));
    TypeProto* gathered_prediction_type_proto = GetGatheredPredictionTypeProto(prediction_arg,
                                                                               graph_defs);
    new_nodes.emplace_back(NodeDef(OpDef{"GatherND", kOnnxDomain, 12},
                                   {ArgDef(prediction_masked_lm), ArgDef("masked_lm_positions_unsqueezed")},
                                   {ArgDef("gathered_prediction", gathered_prediction_type_proto)},
                                   {ONNX_NAMESPACE::MakeAttribute("batch_dims", static_cast<int64_t>(1))},
                                   "GATHERED_LM"));
 
    ONNX_NAMESPACE::TensorProto t_proto;
    t_proto.add_dims(2);
    t_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    t_proto.add_int64_data(static_cast<int64_t>(-1));
    t_proto.add_int64_data(prediction_arg->TypeAsProto()->tensor_type().shape().dim()[2].dim_value());
    new_nodes.emplace_back(NodeDef("Constant",
                                  {},
                                  {ArgDef("logit_reshape", nullptr)},
                                  {ONNX_NAMESPACE::MakeAttribute("value", t_proto)}));

    new_nodes.emplace_back(NodeDef("Reshape",
                                   {ArgDef("gathered_prediction", gathered_prediction_type_proto),
                                    ArgDef("logit_reshape")},                // Inputs
                                   {ArgDef("gathered_prediction_reshaped")}, // Outputs
                                   NodeAttributes(),
                                   "Reshape_gathered_prediction"));

    ONNX_NAMESPACE::TensorProto t_proto_label;
    t_proto_label.add_dims(1);
    t_proto_label.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    t_proto_label.add_int64_data(static_cast<int64_t>(-1));

    new_nodes.emplace_back(NodeDef("Constant",
                                  {},
                                  {ArgDef("label_reshape", nullptr)},
                                  {ONNX_NAMESPACE::MakeAttribute("value", t_proto_label)}));

    new_nodes.emplace_back(NodeDef("Reshape",
                                   {ArgDef("masked_lm_ids", masked_lm_int64_type_proto),
                                    ArgDef("label_reshape")},          // Inputs
                                   {ArgDef("masked_lm_ids_reshaped")}, // Outputs
                                   NodeAttributes(),
                                   "Reshape_label"));

    std::vector<AttributeProto> attrs;
    attrs.push_back(ONNX_NAMESPACE::MakeAttribute("ignore_index", static_cast<int64_t>(0)));
    attrs.push_back(ONNX_NAMESPACE::MakeAttribute("reduction", "mean"));

    new_nodes.emplace_back(NodeDef("SoftmaxCrossEntropyLoss",
                                   {ArgDef("gathered_prediction_reshaped"),
                                    ArgDef("masked_lm_ids_reshaped")},              // Inputs
                                   {ArgDef(mlm_loss, GetLossTypeProto(graph_defs)), // Outputs
                                    ArgDef("probability_lm")},
                                   attrs,
                                   "Masked_LM_Loss"));
  }

  // LabelSoftmaxCrossEntropy for next_sentence
  {
    const NodeArg* ns_prediction_arg = graph.GetNodeArg(prediction_next_sentence);
    ORT_ENFORCE(ns_prediction_arg != nullptr,
                "Next sentence prediction arg ", prediction_next_sentence, " is not found in the graph.");
    TypeProto* next_sentence_labels_type_proto = GetSparseTypeProto(ns_prediction_arg,
                                                                    ONNX_NAMESPACE::TensorProto_DataType_INT64,
                                                                    graph_defs);

    new_nodes.emplace_back(NodeDef("SoftmaxCrossEntropyLoss",
                                   {ArgDef(prediction_next_sentence),
                                    ArgDef(next_sentence_labels, next_sentence_labels_type_proto)}, // Inputs
                                   {ArgDef(nsp_loss, GetLossTypeProto(graph_defs)),
                                    ArgDef("probability_ns", ns_prediction_arg->TypeAsProto())},    // Outputs
                                   {ONNX_NAMESPACE::MakeAttribute("reduction", "mean")},
                                   "Next_Sentence_Loss"));
  }

  // Add
  {
    new_nodes.emplace_back(NodeDef("Add",  // Op
                                   {
                                       ArgDef(mlm_loss),
                                       ArgDef(nsp_loss)  // Inputs
                                   },
                                   {
                                       ArgDef(total_loss, GetLossTypeProto(graph_defs))  // Outputs
                                   },
                                   NodeAttributes(),
                                   "Bert_Total_Loss"  // name
                                   ));
  }

  graph_defs.AddNodeDefs(new_nodes);
  graph_defs.AddGraphInputs({masked_lm_positions, masked_lm_ids, next_sentence_labels});
  graph_defs.AddGraphOutputs({mlm_loss, nsp_loss, total_loss});

  return graph_defs;
}

}  // namespace training
}  // namespace onnxruntime
