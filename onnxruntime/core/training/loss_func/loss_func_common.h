// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include "core/training/graph_augmenter.h"

namespace onnxruntime {
namespace training {

struct LossFunctionInfo {
  //The standard loss function name or an op name (as a cost function)
  std::string name_;

  //The name of the "prediction" of the loss function, must be one of the existing outputs in the model.
  std::string prediction_name_;

  //The name of the "label" of the loss function, must be different from any existing inputs in the model
  std::string label_name_;

  //loss_name Output name of the loss function, must be different from any existing outputs in the model
  std::string loss_name_;
};

class ILossFunction {
 public:
  virtual GraphAugmenter::GraphDefs GetDefs(const LossFunctionInfo& loss_func_info) const = 0;
  virtual ~ILossFunction() {}
};

}  // namespace training
}  // namespace onnxruntime
