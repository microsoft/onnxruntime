// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <unordered_map>
#include <mutex>
#include "core/training/loss_function_builder.h"
#include "core/training/loss_func/mean_squared_error.h"

namespace onnxruntime {
namespace training {

class LossFunctionUsingOperator : public ILossFunction {
 public:
  GraphAugmenter::GraphDefs GetDefs(const LossFunctionInfo& loss_func_info) const {
    GraphAugmenter::GraphDefs graph_defs;
    graph_defs.AddNodeDefs({{
        loss_func_info.name_,
        "",  // node name, leave it empty
        {
            ArgDef(loss_func_info.prediction_name_),  // inputs
            ArgDef(loss_func_info.label_name_)},
        {
            ArgDef(loss_func_info.loss_name_)  // outputs
        }
        // TODO: support setting attributes of the custom op.
    }});

    graph_defs.AddGraphOutputs({loss_func_info.loss_name_});
    return graph_defs;
  }
};

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
}
}  // namespace training
}  // namespace onnxruntime
