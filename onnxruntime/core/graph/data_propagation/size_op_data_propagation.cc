// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "size_op_data_propagation.h"
#include "core/common/common.h"
#include "core/graph/node_arg.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {

Status SizeOpDataPropagation::infer() {
  // Size operator generates a scalar output
  const auto* input_0 = node_.InputDefs()[0];

  // Return and do nothing if input doesn't exist
  if (!input_0 || !input_0->Exists()) {
    return Status::OK();
  }

  if (input_0->GetInferredShapeValues().has_value()) {
    const auto& tensor_shape_proto = input_0->GetInferredShapeValues().value();

    int64_t num_elements = 1;
    // The TensorShapeProto (inferred shape values) should have rank > 0 and
    // all the dimensions have values (not symbolic)
    if (tensor_shape_proto.dim_size() > 0) {
      for (const auto& dim : tensor_shape_proto.dim()) {
        if (!dim.has_dim_value()) {
          return Status::OK();  // Or handle the error appropriately
        }
        num_elements *= dim.dim_value();
      }

      output_def_.SetInferredShapeScalarValue(num_elements);
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime