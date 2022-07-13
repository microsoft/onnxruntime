// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/basic_types.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksTypes.h"

namespace onnxruntime {

class NodeUnit;

namespace nnapi {
namespace gemm_matmul_helpers {

inline bool IsNnapiBatchMatMulAvailable(int32_t nnapi_feature_level) {
  return nnapi_feature_level >= ANEURALNETWORKS_FEATURE_LEVEL_6;
}

// definition in op_support_checker.cc
bool IsGemmOrMatMulSupportedByNnapiFullyConnected(const InitializedTensorSet& initializers,
                                                  const NodeUnit& node_unit,
                                                  int32_t nnapi_feature_level);

}  // namespace gemm_matmul_helpers
}  // namespace nnapi
}  // namespace onnxruntime
