// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/graph/optimizer/adam_optimizer_builder.h"
#include "orttraining/core/graph/optimizer/lamb_optimizer_builder.h"
#include "orttraining/core/graph/optimizer/sgd_optimizer_builder.h"

namespace onnxruntime {
namespace training {

// Register all optimizers here.
void OptimizerBuilderRegistry::RegisterBuilders() {
  GetInstance().Register<AdamOptimizerBuilder>("AdamOptimizer");
  GetInstance().Register<LambOptimizerBuilder>("LambOptimizer");
  GetInstance().Register<SGDOptimizerBuilder>("SGDOptimizer");
}

}  // namespace training
}  // namespace onnxruntime
