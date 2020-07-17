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

TypeProto* BertLoss::GetTransposedTypeProto(const NodeArg* prediction_arg,
                                            GraphAugmenter::GraphDefs& graph_defs) {
  ORT_ENFORCE(prediction_arg != nullptr, "GetTransposedTypeProto's prediction_arg is nullptr");
  const auto* logits_type_proto = prediction_arg->TypeAsProto();
  const auto& dims = logits_type_proto->tensor_type().shape().dim();

  TypeProto* type_proto = graph_defs.CreateTypeProto();
  type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto* target_shape = type_proto->mutable_tensor_type()->mutable_shape();
  // Batch size.
  target_shape->add_dim()->CopyFrom(dims[0]);

  // Class.
  target_shape->add_dim()->CopyFrom(dims[dims.size() - 1]);

  for (int i = 1; i < dims.size() - 1; i++) {
    target_shape->add_dim()->CopyFrom(dims[i]);
  }

  return type_proto;
}

TypeProto* BertLoss::GetGatheredPredictionTransposedTypeProto(const NodeArg* prediction_arg,
                                                              GraphAugmenter::GraphDefs& graph_defs) {
  ORT_ENFORCE(prediction_arg != nullptr, "GetGatheredPredictionTransposedTypeProto's prediction_arg is nullptr");
  const auto* logits_type_proto = prediction_arg->TypeAsProto();
  const auto& dims = logits_type_proto->tensor_type().shape().dim();

  TypeProto* type_proto = graph_defs.CreateTypeProto();
  type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto* target_shape = type_proto->mutable_tensor_type()->mutable_shape();
  // Batch size.
  target_shape->add_dim()->CopyFrom(dims[0]);
  // Vocab size.
  target_shape->add_dim()->CopyFrom(dims[2]);
  // Prediction count.
  target_shape->add_dim()->set_dim_param("dynamic_prediction_count");

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

    // Transpose gathered_predictions with the following permutation: {0, 2, 1} because SoftmaxCrossEntropyLoss accepts
    // scores of the shape {N, C, D1, D2...Dk}.
    TypeProto* gathered_prediction_transposed_type_proto = GetGatheredPredictionTransposedTypeProto(prediction_arg,
                                                                                                    graph_defs);

    new_nodes.emplace_back(NodeDef("Transpose",
                                   {ArgDef("gathered_prediction", gathered_prediction_type_proto)},  // Inputs
                                   {ArgDef("gathered_prediction_transposed",
                                           gathered_prediction_transposed_type_proto)},  // Outputs
                                   {ONNX_NAMESPACE::MakeAttribute("perm", std::vector<int64_t>{static_cast<int64_t>(0),
                                                                                               static_cast<int64_t>(2), static_cast<int64_t>(1)})},

                                   "Transpose_gathered_prediction"));

    std::vector<AttributeProto> attrs;
    attrs.push_back(ONNX_NAMESPACE::MakeAttribute("ignore_index", static_cast<int64_t>(-1)));
    attrs.push_back(ONNX_NAMESPACE::MakeAttribute("reduction", "mean"));

    new_nodes.emplace_back(NodeDef("SoftmaxCrossEntropyLoss",
                                   {ArgDef("gathered_prediction_transposed", gathered_prediction_transposed_type_proto),
                                    ArgDef(masked_lm_ids, masked_lm_int64_type_proto)},  // Inputs
                                   {ArgDef(mlm_loss, GetLossTypeProto(graph_defs)),      // Outputs
                                    ArgDef("probability_lm", gathered_prediction_transposed_type_proto)},
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

    // Transpose prediction_next_sentence with the following permutation: {0, n-1, 1, 2....n-2} because
    // SoftmaxCrossEntropyLoss accepts scores of the shape {N, C, D1, D2...Dk}.

    TypeProto* prediction_next_sentence_transposed_type_proto = GetTransposedTypeProto(ns_prediction_arg, graph_defs);
    const auto* logits_type_proto = ns_prediction_arg->TypeAsProto();
    const auto& dims = logits_type_proto->tensor_type().shape().dim();
    std::vector<int64_t> prediction_next_sentence_transposed_perm;
    prediction_next_sentence_transposed_perm.emplace_back(static_cast<int64_t>(0));
    prediction_next_sentence_transposed_perm.emplace_back(static_cast<int64_t>(dims.size() - 1));

    for (int i = 1; i < dims.size() - 1; i++) {
      prediction_next_sentence_transposed_perm.emplace_back(static_cast<int64_t>(i));
    }

    new_nodes.emplace_back(NodeDef("Transpose",
                                   {ArgDef(prediction_next_sentence)},  // Inputs
                                   {ArgDef("prediction_next_sentence_transposed",
                                           prediction_next_sentence_transposed_type_proto)},  // Outputs
                                   {ONNX_NAMESPACE::MakeAttribute("perm", prediction_next_sentence_transposed_perm)},

                                   "Transpose_prediction_next_sentence"));

    new_nodes.emplace_back(NodeDef("SoftmaxCrossEntropyLoss",
                                   {ArgDef("prediction_next_sentence_transposed", prediction_next_sentence_transposed_type_proto),
                                    ArgDef(next_sentence_labels, next_sentence_labels_type_proto)},  // Inputs
                                   {ArgDef(nsp_loss, GetLossTypeProto(graph_defs)),
                                    ArgDef("probability_ns", prediction_next_sentence_transposed_type_proto)},  // Outputs
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
