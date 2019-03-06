// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <unordered_map>
#include <mutex>
#include "core/training/loss_function_builder.h"
#include "core/training/loss_func/mean_squared_error.h"

namespace onnxruntime {
namespace training {

using namespace std;
using namespace onnxruntime::common;

void LossFunctionRegistry::RegisterCustomLossFunction(const std::string& loss_func_name) {
  ORT_ENFORCE(loss_function_map_.count(loss_func_name) == 0,
              "Failed to register custom loss function, the same name exists:", loss_func_name);

  loss_function_map_[loss_func_name] = [loss_func_name](const LossFunctionInfo& loss_func_info) -> GraphAugmenter::GraphDefs {
    GraphAugmenter::GraphDefs graph_defs;
    graph_defs.AddNodeDefs({{
        loss_func_name,
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
  };
}

void LossFunctionRegistry::RegisterStandardLossFunction(const std::string& loss_func_name,
                                                        const LossFunction& loss_func) {
  ORT_ENFORCE(loss_function_map_.count(loss_func_name) == 0,
              "Failed to register loss function, the same name exists:", loss_func_name);
  loss_function_map_[loss_func_name] = loss_func;
}

const LossFunction* LossFunctionRegistry::GetLossFunction(const std::string& loss_func_name) const {
  auto it = loss_function_map_.find(loss_func_name);
  if (it != loss_function_map_.end()) {
    return &it->second;
  }
  return nullptr;
}

LossFunctionRegistry& LossFunctionRegistry::GetInstance() {
  static LossFunctionRegistry instance;
  return instance;
}

LossFunctionRegistry::LossFunctionRegistry() {
  // Register standard loss functions here.
  RegisterStandardLossFunction("MeanSquaredError", MeanSquaredError);
}

}  // namespace training
}  // namespace onnxruntime
