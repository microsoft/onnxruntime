// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse_sequence_schema_defs.h"

#include "core/graph/constants.h"
#include "core/graph/op.h"
#include <cmath>
#include <type_traits>

namespace onnxruntime {
namespace contrib {

using ::ONNX_NAMESPACE::OPTIONAL;
using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::OpSchema;
using ::ONNX_NAMESPACE::InferenceContext;
// using ::ONNX_NAMESPACE::TensorShapeProto;
// using ::ONNX_NAMESPACE::TensorProto;
// using ::ONNX_NAMESPACE::TensorProto_DataType;


static const char* Reverse_Sequence_ver1_doc = R"DOC(
This op first slices input along the dimension batch_axis, and for each slice i, 
reverses the first seq_lengths[i] elements along the dimension seq_dim.

The elements of seq_lengths must obey seq_lengths[i] <= input.dims[seq_axis], and 
seq_lengths must be a vector of length input.dims[batch_dim].

The output slice i along dimension batch_dim is then given by input slice i, with 
the first seq_lengths[i] slices along dimension seq_dim reversed.
)DOC";


OpSchema& RegisterReverseSequenceOpSchema(OpSchema&& op_schema){
  return op_schema
    .SetDomain(kMSDomain)
    .SinceVersion(1)
    .TypeConstraint(
        "T",
        {"tensor(float)", "tensor(double)", "tensor(int16)", "tensor(int32)", "tensor(int64)" },
        "Constrain input and output types.")
    .TypeConstraint(
        "TIndex",
        {"tensor(int32)", "tensor(int64)" },
        "Constrain input and output types.")
    .Input(
        0,
        "input",
        "The input Tensor([batch, seq, ...] or [seq, batch, ...]) to reverse. ",
        "T")
    .Input(
        1,
        "seq_lengths",
        "1-D with length input.dims(batch_axis) and max(seq_lengths) <= input.dims(seq_axis).",
        "TIndex")
    .Attr(
        "batch_axis",
        "Specify the batch axis of input tensor. Default:0.",
        AttributeProto::INT,
        OPTIONAL)
    .Attr(
        "seq_axis",
        "Specify the batch axis of input tensor. Default:1.",
        AttributeProto::INT,
        OPTIONAL)
    .Output(
        0,
        "Y",
        "The partially reversed input. It has the same shape as input.",
        "T")
    .SetDoc(Reverse_Sequence_ver1_doc)
    .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        propagateShapeFromInputToOutput(ctx, 0, 0);
    });
}

}  // namespace contrib
}  // namespace onnxruntime
