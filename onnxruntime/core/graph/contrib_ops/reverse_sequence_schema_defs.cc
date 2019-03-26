// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse_sequence_schema_defs.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"
#include <cmath>
#include <type_traits>

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
  data_format = "time_major"

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
  data_format = "batch_major"

  output = [[0.0,  1.0,  2.0,  3.0 ],
            [5.0,  4.0,  6.0,  7.0 ],
            [10.0, 9.0,  8.0,  11.0],
            [15.0, 14.0, 13.0, 12.0]]
)DOC";

static const char* Input_Data_Format_ver1_doc = R"DOC(
(Optional) Specify if the input data format is time major (e.g. [seq_length, batch_size, ...]),
or batch major (e.g. [batch_size, seq_length, ...]). Must be one of time_major (default), or batch_major.
)DOC";

void ReverseSequenceShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  auto& first_input_shape = getInputShape(ctx, 0);
  auto rank = first_input_shape.dim_size();
  if (rank < 2) {
    fail_shape_inference("First input tensor must have rank >= 2");
  }

  // auto data_format = getAttribute(ctx, "data_format", "time_major");
  int seq_axis = static_cast<int>(getAttribute(ctx, "seq_axis", 0));
  int batch_axis = static_cast<int>(getAttribute(ctx, "batch_axis", 1));

  if (seq_axis >= rank)
    fail_shape_inference("Invalid seq_axis of ", seq_axis, ". Must be smaller than input rank of ", rank);

  if (batch_axis >= rank)
    fail_shape_inference("Invalid batch_axis of ", batch_axis, ". Must be smaller than input rank of ", rank);

  if (seq_axis == batch_axis)
    fail_shape_inference("seq_axis and batch_axis must have different values but both are ", seq_axis);

  // TensorShapeProto::Dimension batch_size;
  //if (data_format == "time_major") {
  //  batch_size = first_input_shape.dim(1);
  //} else {
  //  batch_size = first_input_shape.dim(0);
  //}
  TensorShapeProto::Dimension batch_dim = first_input_shape.dim(batch_axis);

  if (batch_dim.has_dim_value()) {
    auto& seq_len_input_shape = getInputShape(ctx, 1);
    if (seq_len_input_shape.dim_size() != 1)
      fail_shape_inference("Invalid sequence_lens input. Rank should be 1. Got:", seq_len_input_shape.dim_size());

    auto seq_len_dim0 = seq_len_input_shape.dim(0);
    if (seq_len_dim0.has_dim_value()) {
      if (static_cast<int64_t>(batch_dim.dim_value()) != static_cast<int64_t>(seq_len_dim0.dim_value())) {
        fail_shape_inference("Batch size mismatch for input and sequence_lens. ",
                             batch_dim.dim_value(), " != ", seq_len_dim0.dim_value());
      }
    }
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
      .TypeConstraint(
          "T1", {"tensor(int32)"}, "Constrain sequence_lens to integer tensor.")
      .Attr(
          "seq_axis",
          "The axis containing the sequence dimension. Can be 0 or 1.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "batch_axis",
          "The axis containing the batch dimension. Can be 0 or 1.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Input(
          0,
          "input",
          "Tensor of rank r >= 2, with the shape of `[seq_length, batch_size, ...]` or `[batch_size, seq_length, ...]`",
          "T")
      .Input(
          1,
          "sequence_lens",
          "Tensor specifying lengths of the sequences in a batch. It has shape `[batch_size]`.",
          "T1")
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
