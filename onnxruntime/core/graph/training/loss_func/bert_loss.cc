// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include "core/graph/training/attr_proto_util.h"
#include "core/graph/training/loss_func/bert_loss.h"

namespace onnxruntime {
namespace training {

TypeProto* BertLoss::GetMaskedLMTypeProto(const NodeArg* prediction_arg,
                                          ONNX_NAMESPACE::TensorProto_DataType data_type,
                                          int64_t max_predictions_per_sequence,
                                          GraphAugmenter::GraphDefs& graph_defs) {
  ORT_ENFORCE(prediction_arg != nullptr, "GetMaskedPredictionTypeProto's prediction_arg is nullptr");
  const auto* logits_type_proto = prediction_arg->TypeAsProto();
  const auto& dims = logits_type_proto->tensor_type().shape().dim();

  TypeProto* type_proto = graph_defs.CreateTypeProto();
  type_proto->mutable_tensor_type()->set_elem_type(data_type);

  auto* target_shape = type_proto->mutable_tensor_type()->mutable_shape();
  target_shape->add_dim()->CopyFrom(dims[0]);
  target_shape->add_dim()->set_dim_value(max_predictions_per_sequence);

  return type_proto;
}

TypeProto* BertLoss::GetNSLabelTypeProto(const NodeArg* prediction_arg,
                                         GraphAugmenter::GraphDefs& graph_defs) {
  ORT_ENFORCE(prediction_arg != nullptr, "GetNSLabelTypeProto's prediction_arg is nullptr");
  const auto* logits_type_proto = prediction_arg->TypeAsProto();
  const auto& dims = logits_type_proto->tensor_type().shape().dim();

  TypeProto* type_proto = graph_defs.CreateTypeProto();
  type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);

  auto* target_shape = type_proto->mutable_tensor_type()->mutable_shape();
  target_shape->add_dim()->CopyFrom(dims[0]);

  return type_proto;
}

TypeProto* BertLoss::GetLossTypeProto(GraphAugmenter::GraphDefs& graph_defs) {
  TypeProto* type_proto = graph_defs.CreateTypeProto();
  type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto* target_shape = type_proto->mutable_tensor_type()->mutable_shape();
  target_shape->add_dim()->set_dim_value(1);

  return type_proto;
}

GraphAugmenter::GraphDefs BertLoss::operator()(const Graph& graph, const LossFunctionInfo& loss_func_info) {
  const std::string& total_loss = loss_func_info.loss_name;
  const VectorString& args = loss_func_info.loss_builder_args;
  ORT_ENFORCE(args.size() == 11, " Invalid loss_func_info for BertLoss.");
  const std::string& prediction_masked_lm = args[0];
  const std::string& prediction_next_sentence = args[1];
  const std::string& masked_lm_positions = args[2];
  const std::string& masked_lm_ids = args[3];
  const std::string& masked_lm_weights = args[4];
  const std::string& next_sentence_labels = args[5];
  const std::string& mlm_loss = args[6];
  const std::string& nsp_loss = args[7];
  const int64_t batch_size = static_cast<int64_t>(stoi(args[8]));
  const int64_t max_sequence_len = static_cast<int64_t>(stoi(args[9]));
  const int64_t max_predictions_per_sequence = static_cast<int64_t>(stoi(args[10]));

  std::vector<NodeDef> new_nodes;
  GraphAugmenter::GraphDefs graph_defs;

  // LabelSoftmaxCrossEntropy for masked_lm
  {
    const NodeArg* prediction_arg = graph.GetNodeArg(prediction_masked_lm);
    ORT_ENFORCE(prediction_arg != nullptr,
                "Masked_ML predition arg ", prediction_masked_lm, " is not found in the graph.");
    TypeProto* masked_lm_int64_type_proto = GetMaskedLMTypeProto(prediction_arg,
                                                                 ONNX_NAMESPACE::TensorProto_DataType_INT64,
                                                                 max_predictions_per_sequence,
                                                                 graph_defs);

    ONNX_NAMESPACE::TensorProto shape_tensor_proto;
    shape_tensor_proto.add_dims(2);
    shape_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    shape_tensor_proto.add_int64_data(batch_size);
    shape_tensor_proto.add_int64_data(max_sequence_len);

    new_nodes.emplace_back(NodeDef("Constant",
                                   {},
                                   {ArgDef("label_shape")},
                                   {MakeAttribute("value", shape_tensor_proto)},
                                   "Label_Shape"));

    // Scatter for LM_Label
    {
      ONNX_NAMESPACE::TensorProto zero_tensor_proto;
      zero_tensor_proto.add_dims(1);
      zero_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
      zero_tensor_proto.add_int64_data(int64_t(0));

      new_nodes.emplace_back(NodeDef("ConstantOfShape",
                                     {ArgDef("label_shape")},  // Inputs
                                     {ArgDef("int64_zeros")},  // Outputs
                                     {MakeAttribute("value", zero_tensor_proto)},
                                     "Int64_Zeros"));

      new_nodes.emplace_back(NodeDef("Scatter",
                                     {ArgDef("int64_zeros"),
                                      ArgDef(masked_lm_positions, masked_lm_int64_type_proto),
                                      ArgDef(masked_lm_ids, masked_lm_int64_type_proto)},  // Inputs
                                     {ArgDef("scattered_lm_lables")},                      // Outputs
                                     {MakeAttribute("axis", int64_t(1))},
                                     "Scatter_LM_Lable"));
    }

    //Scatter for LM_Weights
    {
      new_nodes.emplace_back(NodeDef("ConstantOfShape",
                                     {ArgDef("label_shape")},  // Inputs
                                     {ArgDef("float_zeros")},  // Outputs
                                     NodeAttributes(),
                                     "Float_Zeros"));

      TypeProto* masked_lm_float_type_proto = GetMaskedLMTypeProto(prediction_arg,
                                                                   ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                                                                   max_predictions_per_sequence,
                                                                   graph_defs);
      new_nodes.emplace_back(NodeDef("Scatter",
                                     {ArgDef("float_zeros"),
                                      ArgDef(masked_lm_positions, masked_lm_int64_type_proto),
                                      ArgDef(masked_lm_weights, masked_lm_float_type_proto)},  // Inputs
                                     {ArgDef("scattered_lm_weights")},                         // Output
                                     {MakeAttribute("axis", int64_t(1))},
                                     "Scatter_LM_Weights"));
    }

    new_nodes.emplace_back(NodeDef("SparseSoftmaxCrossEntropy",
                                   {ArgDef(prediction_masked_lm),
                                    ArgDef("scattered_lm_lables"),
                                    ArgDef("scattered_lm_weights")},                  // Inputs
                                   {ArgDef(mlm_loss, GetLossTypeProto(graph_defs))},  // Outputs
                                   NodeAttributes(),
                                   "Masked_LM_Loss"));

    graph_defs.AddGraphOutputs({});
  }

  // LabelSoftmaxCrossEntropy for next_sentence
  {
    const NodeArg* ns_prediction_arg = graph.GetNodeArg(prediction_next_sentence);
    ORT_ENFORCE(ns_prediction_arg != nullptr,
                "Next sentence predition arg ", prediction_next_sentence, " is not found in the graph.");
    TypeProto* next_sentence_labels_type_proto = GetNSLabelTypeProto(ns_prediction_arg, graph_defs);

    new_nodes.emplace_back(NodeDef("SparseSoftmaxCrossEntropy",
                                   {ArgDef(prediction_next_sentence),
                                    ArgDef(next_sentence_labels, next_sentence_labels_type_proto)},  // Inputs
                                   {ArgDef(nsp_loss, GetLossTypeProto(graph_defs))},                 // Outputs
                                   NodeAttributes(),
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
  graph_defs.AddGraphOutputs({mlm_loss, nsp_loss, total_loss});

  return graph_defs;
}
}  // namespace training
}  // namespace onnxruntime
