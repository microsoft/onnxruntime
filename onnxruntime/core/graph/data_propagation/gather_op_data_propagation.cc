// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_op_data_propagation.h"
#include "core/common/common.h"
#include "core/graph/node_arg.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/common.h"

namespace onnxruntime {

Status GatherOpDataPropagation::infer() {
  ORT_ENFORCE(output_from_onnx_op_data_propagation_.tensor_type().shape().dim_size() == 0);

  // Following code extracts an element from a 1D array if all conditions are met.
  // e.g.
  // shape data is [1, 3, 64, 64] -> gets 64 if the index is 2.
  // shape data is [1, 3, 64, 64] -> gets 3 if the index is 1.

  // Get "data" input
  // Note: The "data" input should be a one dimension array in this case.
  const auto* input_0 = node_.InputDefs()[0];

  // Get "indices" input
  // Note: The "indices" input should be a scalar, otherwise, if it's a tensor with dimension size > 0,
  //       the operator's type and shape inference should have inferred the output shape value.
  const auto* input_1 = node_.InputDefs()[1];

  // The "indices" should be an initializer because it's a scalar in this case.
  TensorShapeVector indices;
  ORT_RETURN_IF_ERROR(get_initialized_input_values_func_(input_1->Name(), indices));

  // Save the dimension value in the NodeArg.
  // Index value is expected to be within bounds [-s, s-1] along axis of size s
  if (input_0->GetInferredShapeValues().has_value()) {
    const auto& tensor_shape_proto = input_0->GetInferredShapeValues().value();
    auto& dim = tensor_shape_proto.dim(static_cast<int32_t>(HandleNegativeAxis(indices[0], tensor_shape_proto.dim_size())));
    if (dim.has_dim_value()) {
      output_def_.SetInferredShapeScalarValue(dim.dim_value());
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
