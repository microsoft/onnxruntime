// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <unordered_map>
#include <mutex>
#include "loss_function_builder.h"
#include "loss_func/mean_squared_error.h"
#include "loss_func/bert_loss.h"
#include "loss_func/bert_loss_legacy.h"
#include "loss_func/softmax_cross_entropy.h"

namespace onnxruntime {
namespace training {

GraphAugmenter::GraphDefs LossFunctionUsingOperator::operator()(const Graph& graph, const LossFunctionInfo& loss_func_info) {
  const std::string& loss_name = loss_func_info.loss_name;
  const VectorString& args = loss_func_info.loss_builder_args;
  ORT_ENFORCE(args.size() == 2, " Invalid loss_func_info for MeanSquaredError.");
  const std::string& prediction_name = args[0];
  const std::string& label_name = args[1];

  GraphAugmenter::GraphDefs graph_defs;
  std::vector<NodeDef> node_defs;

  const NodeArg* prediction_arg = graph.GetNodeArg(prediction_name);
  ORT_ENFORCE(prediction_arg != nullptr,
              "Prediction arg ", prediction_name, " is not found in the graph. ");
  TypeProto* label_type_proto = graph_defs.CopyTypeProto(prediction_arg);

  node_defs.emplace_back(
      NodeDef(loss_func_info.op_def,  // op_def
              {
                  ArgDef(prediction_name),
                  ArgDef(label_name, label_type_proto)},  // as new graph inputs
              {
                  ArgDef(loss_name)}  // outputs
              // TODO: support setting attributes of the custom op.
              ));

  graph_defs.AddNodeDefs(node_defs);
  graph_defs.AddGraphOutputs({loss_name});

  return graph_defs;
}

void LossFunctionRegistry::RegisterOperatorLossFunction(const std::string& op_name) {
  ORT_ENFORCE(!Contains(op_name),
              "Failed to register loss function using op, the same name exists:", op_name);
  Register<LossFunctionUsingOperator>(op_name,
                                      []() -> std::unique_ptr<LossFunctionUsingOperator> {
                                        return std::make_unique<LossFunctionUsingOperator>();
                                      });
}

#define REGISTER_NON_OPERATOR_LOSS_FUNCTION(func) LossFunctionRegistry::GetInstance().Register<func>(#func);

void LossFunctionRegistry::RegisterNonOperatorLossFunctions() {
  // Register non-operator loss functions here.
  REGISTER_NON_OPERATOR_LOSS_FUNCTION(MeanSquaredError);
  REGISTER_NON_OPERATOR_LOSS_FUNCTION(BertLoss);
  REGISTER_NON_OPERATOR_LOSS_FUNCTION(BertLossLegacy);
  REGISTER_NON_OPERATOR_LOSS_FUNCTION(SoftmaxCrossEntropy);
  REGISTER_NON_OPERATOR_LOSS_FUNCTION(SparseSoftmaxCrossEntropy);
  REGISTER_NON_OPERATOR_LOSS_FUNCTION(SoftmaxCrossEntropyLoss);
}
}  // namespace training
}  // namespace onnxruntime
