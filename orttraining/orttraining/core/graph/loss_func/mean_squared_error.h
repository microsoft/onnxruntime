// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include "orttraining/core/graph/loss_func/loss_func_common.h"

namespace onnxruntime {
namespace training {

struct MeanSquaredError : public ILossFunction {
  GraphAugmenter::GraphDefs operator()(const Graph& graph, const LossFunctionInfo& loss_func_info) override;
};

}  // namespace training
}  // namespace onnxruntime
