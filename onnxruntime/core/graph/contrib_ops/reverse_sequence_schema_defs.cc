// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse_sequence_schema_defs.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"
#include <cmath>
#include <type_traits>

namespace onnxruntime {
namespace contrib {

using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::OPTIONAL;
using ::ONNX_NAMESPACE::OpSchema;
using ::ONNX_NAMESPACE::InferenceContext;
using ::ONNX_NAMESPACE::TensorShapeProto;
using ::ONNX_NAMESPACE::TensorProto;
using ::ONNX_NAMESPACE::TensorProto_DataType;

static const char* ReverseSequence_ver1_doc = R"DOC(
Reverse batch of sequences having different lengths specified by `sequence_lens`.

For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis.

Example 1:
  input = [
    [0.0, 1.0, 2.0],
    [3.0, 4.0, 5.0],
  ]
  sequence_lens = [3, 3]
  data_format = "batch_major"

  output = [
    [2.0, 1.0, 0.0],
    [5.0, 4.0, 3.0],
  ]

Example 2:
  input = [
    [0.0, 1.0, 2.0],
    [3.0, 4.0, 5.0],
  ]
  sequence_lens = [2, 1]
  data_format = "batch_major"

  output = [
    [1.0, 0.0, 2.0],
    [3.0, 4.0, 5.0],
  ]
)DOC";

static const char* Input_Data_Format_ver1_doc = R"DOC(
(Optional) Specify if the input data format is time major (e.g. [seq_length, batch_size, ...]),
or batch major (e.g. [batch_size, seq_length, ...]). Must be one of time_major (default), or batch_major.
)DOC";

OpSchema& RegisterReverseSequenceOpSchema(OpSchema&& op_schema) {
  return op_schema
    .SetDomain(kMSDomain)
    .SinceVersion(1)
    .TypeConstraint(
      "T",
      OpSchema::all_tensor_types(),
      "Input and output types can be of any tensor type.")
    .TypeConstraint(
      "T1", {"tensor(int32)"}, "Constrain seq_lens to integer tensor.")
    .Attr(
      "data_format",
      Input_Data_Format_ver1_doc,
      AttributeProto::STRING,
      std::string("time_major"))
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
    .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);
}

}  // namespace contrib
}  // namespace onnxruntime
