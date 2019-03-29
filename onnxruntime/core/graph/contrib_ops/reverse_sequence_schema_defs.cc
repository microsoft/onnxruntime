// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse_sequence_schema_defs.h"
#include "core/graph/op.h"

namespace onnxruntime {
namespace contrib {

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::InferenceContext;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL;
using ONNX_NAMESPACE::TensorProto;
using ONNX_NAMESPACE::TensorProto_DataType;
using ONNX_NAMESPACE::TensorShapeProto;

static const char* ReverseSequence_ver1_doc = R"DOC(
Reverse batch of sequences having different lengths specified by `sequence_lens`.

For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.

Example 1:
  input = [[0.0, 4.0, 8.0,  12.0],
           [1.0, 5.0, 9.0,  13.0],
           [2.0, 6.0, 10.0, 14.0],
           [3.0, 7.0, 11.0, 15.0]]
  sequence_lens = [4, 3, 2, 1]
  time_axis = 0
  batch_axis = 1

  output = [[3.0, 6.0, 9.0,  12.0],
            [2.0, 5.0, 8.0,  13.0],
            [1.0, 4.0, 10.0, 14.0],
            [0.0, 7.0, 11.0, 15.0]]

Example 2:
  input = [[0.0,  1.0,  2.0,  3.0 ],
           [4.0,  5.0,  6.0,  7.0 ],
           [8.0,  9.0,  10.0, 11.0],
           [12.0, 13.0, 14.0, 15.0]]
  sequence_lens = [1, 2, 3, 4]
  time_axis = 1
  batch_axis = 0

  output = [[0.0,  1.0,  2.0,  3.0 ],
            [5.0,  4.0,  6.0,  7.0 ],
            [10.0, 9.0,  8.0,  11.0],
            [15.0, 14.0, 13.0, 12.0]]
)DOC";

void ReverseSequenceShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  auto& first_input_shape = getInputShape(ctx, 0);
  if (first_input_shape.dim_size() < 2) {
    fail_shape_inference("'input' must have rank >= 2");
  }
  auto& seq_len_input_shape = getInputShape(ctx, 1);
  if (seq_len_input_shape.dim_size() != 1) {
    fail_shape_inference("'sequence_lens' must have rank of 1");
  }

  propagateShapeFromInputToOutput(ctx, 0, 0);
}

OpSchema& RegisterReverseSequenceOpSchema(OpSchema&& op_schema) {
  return op_schema
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types(),
          "Input and output types can be of any tensor type.")
      .Attr(
          "time_axis",
          "(Optional) Specify which axis is time axis. Must be one of 0 (default), or 1.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "batch_axis",
          "(Optional) Specify which axis is batch axis. Must be one of 1 (default), or 0.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Input(
          0,
          "input",
          "Tensor of rank r >= 2.",
          "T")
      .Input(
          1,
          "sequence_lens",
          "Tensor specifying lengths of the sequences in a batch. It has shape `[batch_size]`.",
          "tensor(int32)")
      .Output(
          0,
          "Y",
          "Tensor with same shape of input.",
          "T")
      .SetDoc(ReverseSequence_ver1_doc)
      .TypeAndShapeInferenceFunction(ReverseSequenceShapeInference);
}

}  // namespace contrib
}  // namespace onnxruntime
