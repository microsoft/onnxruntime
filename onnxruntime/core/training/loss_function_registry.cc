// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <unordered_map>
#include <mutex>
#include "core/training/loss_function_builder.h"
#include "core/training/loss_func/mean_squared_error.h"

namespace onnxruntime {
namespace training {

class CustomLossFunction : public ILossFunction {
 public:
  CustomLossFunction(const std::string& loss_func_name) : loss_func_name_(loss_func_name) {
  }

  GraphAugmenter::GraphDefs GetDefs(const LossFunctionInfo& loss_func_info) const {
    GraphAugmenter::GraphDefs graph_defs;
    graph_defs.AddNodeDefs({{
        loss_func_name_,
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

 private:
  const std::string loss_func_name_;
};

void LossFunctionRegistry::RegisterCustomLossFunction(const std::string& loss_func_name) {
  ORT_ENFORCE(MakeUnique(loss_func_name) == nullptr,
              "Failed to register custom loss function, the same name exists:", loss_func_name);
  Register<CustomLossFunction>(loss_func_name,
                               [loss_func_name]() -> std::unique_ptr<CustomLossFunction> {
                                 return std::make_unique<CustomLossFunction>(loss_func_name);
                               });
}

#define REGISTER_LOSS_FUNCTION(func) LossFunctionRegistry::GetInstance().Register<func>(#func);

void LossFunctionRegistry::RegisterStandardLossFunctions() {
  // Register standard loss functions here.
  REGISTER_LOSS_FUNCTION(MeanSquaredError);
}
}  // namespace training
}  // namespace onnxruntime
