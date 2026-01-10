// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sub_op_data_propagation.h"
#include "core/common/common.h"
#include "core/graph/node_arg.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/common.h"

namespace onnxruntime {

Status SubOpDataPropagation::infer() {
  // Get "A" input
  const auto* input_0 = node_.InputDefs()[0];
  // Get "B" input
  const auto* input_1 = node_.InputDefs()[1];

  // Return and do nothing if input doesn't exist
  if (!input_0 || !input_1 || !input_0->Exists() || !input_1->Exists()) {
    return Status::OK();
  }

  if (input_0->GetInferredShapeScalarValue().has_value() && input_1->GetInferredShapeScalarValue().has_value()) {
    output_def_.SetInferredShapeScalarValue(
        input_0->GetInferredShapeScalarValue().value() -
        input_1->GetInferredShapeScalarValue().value());
  }

  return Status::OK();
}

}  // namespace onnxruntime
