// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "select_schema_defs.h"

#include "core/graph/constants.h"
// #include "core/graph/op.h"

using ONNX_NAMESPACE::OPTIONAL;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::InferenceContext;
using ONNX_NAMESPACE::TensorShapeProto;
using onnxruntime::kMSDomain;

namespace onnxruntime {
namespace contrib {

static const char* Select_ver1_doc = R"DOC(
Select element or Row from X or Y according to condition.

X and Y must be of same shape. The condition tensor must be a scalar if X and Y are scalar. 
If X and Y are vectors of higher rank, then condition must be either a vector with size 
matching the first dimension of X, or must have the same shape as X.

The condition tensor acts as a mask that chooses, based on the value at each element, 
whether the corresponding element / row in the output should be taken from X (if true) 
or Y (if false).

If condition is a vector and X and Y are higher rank matrices, then it chooses which 
row (outer dimension) to copy from X and Y. If condition has the same shape as X and Y, 
then it chooses which element to copy from X and Y.
)DOC";


OpSchema& RegisterSelectOpSchema(OpSchema&& op_schema) {
  return op_schema
    .SetDomain(kMSDomain)
    .SinceVersion(1)
    .TypeConstraint(
        "T",
        OpSchema::all_numeric_types(),
        "Constrain X, Y and output types.")
    .TypeConstraint(
        "TCond",
        {"tensor(bool)"},
        "Constrain condition types.")
    .Input(
        0,
        "condition",
        "Tensor, either same shape as X, or 1d vector with depth same as X.dims[0]",
        "TCond")
    .Input(
        1,
        "X",
        "Tensor whose value or row will be used when correspond condition is true.",
        "T")
    .Input(
        2,
        "Y",
        "Tensor whose value or row will be used when correspond condition is false. Must be of same shape as X.",
        "T")
    .Output(
        0,
        "Z",
        "result tensor. same shape as X.",
        "T")
    .SetDoc(Select_ver1_doc)
    .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
            propagateElemTypeFromInputToOutput(ctx, 1, 0);
            propagateShapeFromInputToOutput(ctx, 1, 0);
        });
}

}  // namespace contrib
}  // namespace onnxruntime