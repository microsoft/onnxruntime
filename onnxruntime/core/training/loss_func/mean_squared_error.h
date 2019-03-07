// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include "core/training/loss_func/loss_func_common.h"

namespace onnxruntime {
namespace training {

class MeanSquaredError : public ILossFunction {
 public:
  GraphAugmenter::GraphDefs GetDefs(const LossFunctionInfo& loss_func_info) const override;
};

}  // namespace training
}  // namespace onnxruntime
