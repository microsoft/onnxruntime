// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "onnx/onnx_pb.h"

namespace onnxruntime {
namespace internal_nhwc_onnx {
using ONNX_NAMESPACE::AttributeProto;

void RegisterInternalNHWCOpset() {
ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearSoftmax)
    .SinceVersion(1)
    .SetDomain(onnxruntime::kMSInternalNHWCDomain)
    .SetDoc(R"DOC(
QLinearSoftmax computes the normalized exponential values for the given input:

Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)

The input does not need to explicitly be a 2D vector. The "axis" attribute
indicates the dimension along which QLinearSoftmax will be performed.
The output tensor has the same shape
and contains the QLinearSoftmax values of the corresponding input.
)DOC")
                             .Attr("opset", "use to mark SinceVersion", AttributeProto::INT, static_cast<int64_t>(1))
                             .Attr("axis", "use to mark SinceVersion", AttributeProto::INT, static_cast<int64_t>(-1))
                             .Input(0, "X",
                                    "The input tensor that's coerced into a 2D matrix of size (NxD)",
                                    "T")
                             .Input(1, "x_scale", "Scale of quantized input 'X'. It must be a scalar.",
                                    "tensor(float)")
                             .Input(2, "x_zero_point",
                                    "Zero point tensor for input 'X'."
                                    "It must be a scalar.",
                                    "T")
                             .Input(3, "y_scale", "Scale of quantized output 'Y'. It must be a scalar.",
                                    "tensor(float)")
                             .Input(4, "y_zero_point",
                                    "Zero point tensor for output 'Y'. "
                                    "It must be a scalar.",
                                    "T")
                             .Output(0, "Y",
                                     "Output data tensor from pooling across the input "
                                     "tensor. The output tensor has the same rank as the input. "
                                     "with the N and C value keep it value, while the other"
                                     "dimensions are all 1.",
                                     "T")
                             .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                                             "Constrain input and output types to singed/unsigned int8 tensors.")
                             .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                               // Type inference
                               propagateElemTypeFromInputToOutput(ctx, 0, 0);

                               // Shape inference starts
                               if (!hasNInputShapes(ctx, 1)) {
                                 return;
                               }

                               // Validate the value of 'axis'
                               const ONNX_NAMESPACE::TensorShapeProto& input_shape =
                                   ctx.getInputType(0)->tensor_type().shape();
                               int r = input_shape.dim_size();
                               int axis = static_cast<int>(getAttribute(ctx, "axis", -1));
                               if (axis < -r || axis >= r) {
                                 fail_shape_inference(
                                     "'axis' must be in [",
                                     -r,
                                     " , ",
                                     (r - 1),
                                     "]. Its actual value is: ",
                                     axis);
                               }

                               // Shape inference
                               propagateShapeFromInputToOutput(ctx, 0, 0);
                             });
}
}  // namespace internal_nhwc_onnx
}  // namespace onnxruntime
