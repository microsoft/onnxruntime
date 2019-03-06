// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include "core/training/loss_func/loss_func_common.h"
#include "core/training/graph_augmenter.h"

namespace onnxruntime {
namespace training {

GraphAugmenter::GraphDefs MeanSquaredError(const LossFunctionInfo& loss_func_info);

}  // namespace training
}  // namespace onnxruntime
