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

Status IsMatchingTypeAndShape(
    const onnxruntime::Tensor& tensor,
    const int32_t element_type,
    const std::vector<int64_t>& expected_shape_dims) {
  ORT_RETURN_IF_NOT(tensor.GetElementType() == element_type, "Type mismatch");
  const TensorShape& tensor_shape = tensor.Shape();
  TensorShape expected_shape(expected_shape_dims);
  ORT_RETURN_IF_NOT(tensor_shape == expected_shape, "Mismatch: expected:[", tensor_shape, "], actual:[", expected_shape, "]");
  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
