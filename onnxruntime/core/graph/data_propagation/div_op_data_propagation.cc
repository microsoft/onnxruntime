// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "div_op_data_propagation.h"
#include "core/common/common.h"
#include "core/graph/node_arg.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/common.h"
#include "core/graph/data_propagation/data_propagation_value_utils.h"

namespace onnxruntime {

Status DivOpDataPropagation::infer() {
  // Get "A" input
  const auto* input_0 = node_.InputDefs()[0];
  // Get "B" input
  const auto* input_1 = node_.InputDefs()[1];

  // Return and do nothing if input doesn't exist
  if (!input_0 || !input_1 || !input_0->Exists() || !input_1->Exists()) {
    return Status::OK();
  }

  int64_t lhs = 0;
  int64_t rhs = 0;
  bool lhs_is_rank1 = false;
  bool rhs_is_rank1 = false;
  if (TryGetSinglePropagatedShapeValue(*input_0, lhs, lhs_is_rank1) &&
      TryGetSinglePropagatedShapeValue(*input_1, rhs, rhs_is_rank1) &&
      rhs != 0) {
    // Single-element operands may be carried as a rank-0 scalar or a rank-1 [1] value. Per ONNX
    // broadcasting, the result is rank-1 if either operand is rank-1, otherwise a scalar; keep
    // the propagated value's rank consistent with that so downstream consumers see the right rank.
    SetSinglePropagatedShapeValue(output_def_, lhs / rhs, lhs_is_rank1 || rhs_is_rank1);
  }

  return Status::OK();
}

}  // namespace onnxruntime
