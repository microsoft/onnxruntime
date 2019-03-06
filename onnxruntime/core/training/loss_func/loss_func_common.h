// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>

namespace onnxruntime {
namespace training {

struct LossFunctionInfo {
  //The standard loss function name or a custom op name as a cost function
  std::string loss_func_name_;

  //The name of the "prediction" of the loss function, must be one of the existing outputs in the model.
  std::string prediction_name_;

  //The name of the "label" of the loss function, must be different from any existing inputs in the model
  std::string label_name_;

  //loss_name Output name of the loss function, must be different from any existing outputs in the model
  std::string loss_name_;
};

//TODO: there will be an interface class for all loss funcs.

}  // namespace training
}  // namespace onnxruntime
