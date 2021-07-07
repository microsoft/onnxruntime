// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include "orttraining/core/graph/loss_func/bert_loss_legacy.h"
#include "onnx/defs/attr_proto_util.h"
#include "onnx/defs/tensor_proto_util.h"

namespace onnxruntime {
namespace training {

TypeProto* BertLossLegacy::GetMaskedLMTypeProto(const NodeArg* prediction_arg,
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

TypeProto* BertLossLegacy::GetGatheredPredictionTypeProto(const NodeArg* prediction_arg,
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

TypeProto* BertLossLegacy::GetLossTypeProto(GraphAugmenter::GraphDefs& graph_defs) {
  return graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
}

GraphAugmenter::GraphDefs BertLossLegacy::operator()(const Graph& graph, const LossFunctionInfo& loss_func_info) {
  const std::string& total_loss = loss_func_info.loss_name;
  const VectorString& args = loss_func_info.loss_builder_args;
  ORT_ENFORCE(args.size() == 8, " Invalid loss_func_info for BertLoss.");
  const std::string& prediction_masked_lm = args[0];
  const std::string& prediction_next_sentence = args[1];
  const std::string& masked_lm_positions = args[2];
  const std::string& masked_lm_ids = args[3];
  const std::string& masked_lm_weights = args[4];
  const std::string& next_sentence_labels = args[5];
  const std::string& mlm_loss = args[6];
  const std::string& nsp_loss = args[7];

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
    int onnx_opset = -1;
    auto onnx_domain_it = graph.DomainToVersionMap().find(kOnnxDomain);
    if (onnx_domain_it != graph.DomainToVersionMap().end()) {
      onnx_opset = onnx_domain_it->second;
    } else {
      auto onnx_domain_alias_it = graph.DomainToVersionMap().find(kOnnxDomainAlias);
      if (onnx_domain_alias_it != graph.DomainToVersionMap().end())
        onnx_opset = onnx_domain_alias_it->second;
      else
        ORT_THROW("ONNX domain not found in this model");
    }
    
    if (onnx_opset <= 12) {
      new_nodes.emplace_back(NodeDef("Unsqueeze",
                                     {ArgDef(masked_lm_positions, masked_lm_int64_type_proto)},
                                     {ArgDef("masked_lm_positions_unsqueezed")},
                                     {ONNX_NAMESPACE::MakeAttribute("axes", std::vector<int64_t>{static_cast<int64_t>(2)})},
                                     "Mask_LM_Positions_Unsqueezed"));
    } else {
      auto t_proto = ONNX_NAMESPACE::ToTensor<int64_t>(1);
      TypeProto* int64_t_proto = graph_defs.CreateTypeProto();
      int64_t_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64);

      const std::string two_constant = "bert_legacy_two_constant";
      new_nodes.emplace_back(NodeDef("Constant",
                                     {},
                                     {ArgDef(two_constant, int64_t_proto)},
                                     {ONNX_NAMESPACE::MakeAttribute("value", t_proto)}));
      new_nodes.emplace_back(NodeDef("Unsqueeze",
                                     {ArgDef(masked_lm_positions, masked_lm_int64_type_proto),
                                      ArgDef("two_constant", int64_t_proto)},
                                     {ArgDef("masked_lm_positions_unsqueezed")},
                                     NodeAttributes(),
                                     "Mask_LM_Positions_Unsqueezed"));
    }
    
    TypeProto* gathered_prediction_type_proto = GetGatheredPredictionTypeProto(prediction_arg,
                                                                               graph_defs);
    new_nodes.emplace_back(NodeDef(OpDef{"GatherND", kOnnxDomain, 12},
                                   {ArgDef(prediction_masked_lm), ArgDef("masked_lm_positions_unsqueezed")},
                                   {ArgDef("gathered_prediction", gathered_prediction_type_proto)},
                                   {ONNX_NAMESPACE::MakeAttribute("batch_dims", static_cast<int64_t>(1))},
                                   "GATHERED_LM"));

    TypeProto* masked_lm_float_type_proto = GetMaskedLMTypeProto(prediction_arg,
                                                                 ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                                                                 graph_defs);
    new_nodes.emplace_back(NodeDef("SparseSoftmaxCrossEntropy",
                                   {ArgDef("gathered_prediction", gathered_prediction_type_proto),
                                    ArgDef(masked_lm_ids, masked_lm_int64_type_proto),
                                    ArgDef(masked_lm_weights, masked_lm_float_type_proto)},  // Inputs
                                   {ArgDef(mlm_loss, GetLossTypeProto(graph_defs)),          // Outputs
                                    ArgDef("probability_lm", gathered_prediction_type_proto)},
                                   {ONNX_NAMESPACE::MakeAttribute("reduction", "mean")},
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

    new_nodes.emplace_back(NodeDef("SparseSoftmaxCrossEntropy",
                                   {ArgDef(prediction_next_sentence),
                                    ArgDef(next_sentence_labels, next_sentence_labels_type_proto)},  // Inputs
                                   {ArgDef(nsp_loss, GetLossTypeProto(graph_defs)),
                                    ArgDef("probability_ns", ns_prediction_arg->TypeAsProto())},  // Outputs
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
  graph_defs.AddGraphInputs({masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels});
  graph_defs.AddGraphOutputs({mlm_loss, nsp_loss, total_loss});

  return graph_defs;
}

}  // namespace training
}  // namespace onnxruntime
