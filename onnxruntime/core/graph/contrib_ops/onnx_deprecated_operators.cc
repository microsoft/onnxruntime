// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"

// Suppress a warning: global initializer calls a non-constexpr function 'symbol' which is from
// ONNX_OPERATOR_SET_SCHEMA_EX macro and only happens in debug build
#if defined(_WIN32) && !defined(NDEBUG)
#pragma warning(disable : 26426)
#endif

// Register removed experimental ops for backward compatibility.
// Experimental operators do not have version history. However, Windows 10 1809(RS5) takes bunch of experimental operators
// as production ops. In order to maintain backward compatibility when the experimental ops are removed from ONNX
// they need to be added in onnxruntime as contrib ops.
// ONNX exp ops(Affine, Crop, ParametricSoftplus, ImageScaler, ThresholdedRelu, DynamicSlice, ScaledTanh, MVN) old
// version history maintenance
// See: https://github.com/onnx/onnx/pull/1909

#include "core/graph/contrib_ops/contrib_defs.h"
using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace contrib {
constexpr const char* Affine_ver1_doc = R"DOC(
Affine takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the affine function, y = alpha * x + beta,
is applied to the tensor elementwise.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Affine, 1,
    OpSchema()
        .SetDoc(Affine_ver1_doc)
        .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, 1.0f)
        .Attr("beta", "Value of beta", AttributeProto::FLOAT, 0.0f)
        .Input(0, "X", "1D input tensor", "T")
        .Output(0, "Y", "1D output tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* ParametricSoftplus_ver1_doc = R"DOC(
ParametricSoftplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = alpha * ln(exp(beta * x) + 1), is applied to
the tensor elementwise.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ParametricSoftplus, 1,
    OpSchema()
        .SetDoc(ParametricSoftplus_ver1_doc)
        .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("beta", "Value of beta", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Input(0, "X", "1D input tensor", "T")
        .Output(0, "Y", "1D input tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* ImageScaler_ver1_doc =
    R"DOC(Scale and bias the input image. Bias values are stored in
the same ordering as the image pixel format.)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ImageScaler, 1,
    OpSchema()
        .SetDoc(ImageScaler_ver1_doc)
        .Attr("bias", "Bias applied to each channel, same size as C.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(0, "output", "Result, has same shape and type as input", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* Crop_ver1_doc =
    R"DOC(Crop and image to the specified spatial dimensions. If scale is given,
then optionally start the crop offset by the left/top border amounts.
If scale is not provided, crop the borders as provided.)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Crop, 1,
    OpSchema()
        .SetDoc(Crop_ver1_doc)
        .Attr("border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).", AttributeProto::INTS,
              OPTIONAL_VALUE)
        .Attr("scale", "A 1-D values of (height, width).", AttributeProto::INTS, OPTIONAL_VALUE)
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(0, "output", "Result, has same type as input, with H and W dimensions reduced.", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors."));

constexpr const char* ThresholdedRelu_ver1_doc = R"DOC(
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise. )DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ThresholdedRelu, 1,
    OpSchema()
        .SetDoc(ThresholdedRelu_ver1_doc)
        .Attr("alpha", "Threshold value", AttributeProto::FLOAT, 1.0f)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* DynamicSlice_ver1_doc = R"DOC(
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `axes`, `starts` and `ends` inputs to specify the start and end
dimension for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  result = [
      [5, 6, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    DynamicSlice, 1,
    OpSchema()
        .SetDoc(DynamicSlice_ver1_doc)
        .Input(0, "data", "Tensor of data to extract slices from.", "T")
        .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
        .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes", "Tind")
        .Input(3, "axes", "1-D tensor of axes that `starts` and `ends` apply to.", "Tind", OpSchema::Optional)
        .Output(0, "output", "Sliced data tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types"));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(GivenTensorFill, 1,
                                 OpSchema()
                                     .Input(0, "shape", "The shape of filled tensor", "T", OpSchema::Optional)
                                     .Output(0, "X", "The filled tensor", "T")
                                     .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                                     "Constrain input and output types to float tensors.")
                                     .Attr("values", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
                                     .Attr("shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
                                     .Attr("input_as_shape", "", AttributeProto::INT, OPTIONAL_VALUE)
                                     .Attr("extra_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
                                     .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                       ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                       if (ctx.getAttribute("shape") != nullptr) {
                                         propagateShapeFromAttributeToOutput(ctx, "shape", 0);
                                         return;
                                       }
                                       // The type constraints above do not allow for input_as_shape
                                       // and may need to be fixed.
                                       if (getAttribute(ctx, "input_as_shape", 0) != 0)  // dynamic shape
                                         return;
                                       std::vector<int64_t> extra_shape;
                                       getRepeatedAttribute(ctx, "extra_shape", extra_shape);
                                       if (hasInputShape(ctx, 0)) {
                                         ONNX_NAMESPACE::TensorShapeProto shape =
                                             ctx.getInputType(0)->tensor_type().shape();
                                         for (auto extra_dim_val : extra_shape) {
                                           if (extra_dim_val < 0)
                                             fail_shape_inference(
                                                 "Negative values are not allowed in a shape specification");
                                           shape.add_dim()->set_dim_value(extra_dim_val);
                                         }
                                         updateOutputShape(ctx, 0, shape);
                                       }
                                     }));

constexpr const char* Scale_ver1_doc = R"DOC(
Scale takes one input data (Tensor<float>) and produces one output data
(Tensor<float>) whose value is the input data tensor scaled element-wise.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Scale, 1,
    OpSchema()
        .Input(0, "input", "Input data to be scaled", "T")
        .Output(0, "output", "Output data after scaling", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .SetDoc(Scale_ver1_doc)
        .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* GRUUnit_ver1_doc = R"DOC(
GRUUnit computes the activations of a standard GRU,
in a sequence-length aware fashion.
Concretely, given the (fused) inputs X (TxNxD), the previous hidden
state (NxD), and the sequence lengths (N), computes the GRU
activations, avoiding computation if the input is invalid (as in, the
value at X[t][n] >= seqLengths[n].
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(GRUUnit, 1,
                                 OpSchema()
                                     .SetDoc(GRUUnit_ver1_doc)
                                     .Attr("drop_states",
                                           "Bool to determine if hidden state is zeroes or passed "
                                           "along for timesteps past the given sequence_length.",
                                           AttributeProto::INT, OPTIONAL_VALUE)
                                     .Input(0, "hidden_prev", "The previous GRU hidden state.", "T")
                                     .Input(1, "gates",
                                            "Unactivated gate outputs from forget, update, "
                                            "and output gates, pre-activation.",
                                            "T")
                                     .Input(2, "seq_lengths",
                                            "Array of sequence lengths.  "
                                            "len(seq_lengths) should equal batch size N.",
                                            "T")
                                     .Input(3, "t", "The timestep for this operation.", "T")
                                     .Output(0, "hidden", "The new GRU hidden state calculated by this op.", "T")
                                     .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                                     "Constrain input and output types to float tensors."));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(GivenTensorFill, 10,
                                 OpSchema()
                                     .Deprecate()
                                     .Input(0, "shape", "The shape of filled tensor", "T", OpSchema::Optional)
                                     .Output(0, "X", "The filled tensor", "T")
                                     .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                                     "Constrain input and output types to float tensors.")
                                     .Attr("values", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
                                     .Attr("shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
                                     .Attr("input_as_shape", "", AttributeProto::INT, OPTIONAL_VALUE)
                                     .Attr("extra_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
                                     .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                       ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                       if (ctx.getAttribute("shape") != nullptr) {
                                         propagateShapeFromAttributeToOutput(ctx, "shape", 0);
                                         return;
                                       }
                                       // The type constraints above do not allow for input_as_shape
                                       // and may need to be fixed.
                                       if (getAttribute(ctx, "input_as_shape", 0) != 0)  // dynamic shape
                                         return;
                                       std::vector<int64_t> extra_shape;
                                       getRepeatedAttribute(ctx, "extra_shape", extra_shape);
                                       if (hasInputShape(ctx, 0)) {
                                         ONNX_NAMESPACE::TensorShapeProto shape =
                                             ctx.getInputType(0)->tensor_type().shape();
                                         for (auto extra_dim_val : extra_shape) {
                                           if (extra_dim_val < 0)
                                             fail_shape_inference(
                                                 "Negative values are not allowed in a shape specification");
                                           shape.add_dim()->set_dim_value(extra_dim_val);
                                         }
                                         updateOutputShape(ctx, 0, shape);
                                       }
                                     }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Scale, 10,
    OpSchema()
        .Deprecate()
        .Input(0, "input", "Input data to be scaled", "T")
        .Output(0, "output", "Output data after scaling", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .SetDoc(Scale_ver1_doc)
        .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(GRUUnit, 10,
                                 OpSchema()
                                     .Deprecate()
                                     .SetDoc(GRUUnit_ver1_doc)
                                     .Attr("drop_states",
                                           "Bool to determine if hidden state is zeroes or passed "
                                           "along for timesteps past the given sequence_length.",
                                           AttributeProto::INT, OPTIONAL_VALUE)
                                     .Input(0, "hidden_prev", "The previous GRU hidden state.", "T")
                                     .Input(1, "gates",
                                            "Unactivated gate outputs from forget, update, "
                                            "and output gates, pre-activation.",
                                            "T")
                                     .Input(2, "seq_lengths",
                                            "Array of sequence lengths.  "
                                            "len(seq_lengths) should equal batch size N.",
                                            "T")
                                     .Input(3, "t", "The timestep for this operation.", "T")
                                     .Output(0, "hidden", "The new GRU hidden state calculated by this op.", "T")
                                     .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                                     "Constrain input and output types to float tensors."));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    MeanVarianceNormalization, 1,
    OpSchema()
        .SetDoc(R"DOC(Perform mean variance normalization.)DOC")
        .Attr("across_channels", "If 1, mean and variance are computed across channels. Default is 0.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("normalize_variance", "If 0, normalize the mean only.  Default is 1.", AttributeProto::INT,
              static_cast<int64_t>(1))
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(0, "output", "Result, has same shape and type as input", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ScaledTanh, 1,
    OpSchema()
        .Attr("alpha", "Scaling value", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("beta", "Scaling value", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Input(0, "input", "Input tensor", "T")
        .Output(0, "output",
                "The scaled hyperbolic tangent values of the input tensor "
                "computed element-wise",
                "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Affine, 10,
    OpSchema()
        .Deprecate()
        .SetDoc(Affine_ver1_doc)
        .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, 1.0f)
        .Attr("beta", "Value of beta", AttributeProto::FLOAT, 0.0f)
        .Input(0, "X", "1D input tensor", "T")
        .Output(0, "Y", "1D output tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ParametricSoftplus, 10,
    OpSchema()
        .Deprecate()
        .SetDoc(ParametricSoftplus_ver1_doc)
        .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("beta", "Value of beta", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Input(0, "X", "1D input tensor", "T")
        .Output(0, "Y", "1D input tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ImageScaler, 10,
    OpSchema()
        .Deprecate()
        .SetDoc(ImageScaler_ver1_doc)
        .Attr("bias", "Bias applied to each channel, same size as C.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(0, "output", "Result, has same shape and type as input", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Crop, 10,
    OpSchema()
        .Deprecate()
        .SetDoc(Crop_ver1_doc)
        .Attr("border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).", AttributeProto::INTS)
        .Attr("scale", "A 1-D values of (height, width).", AttributeProto::INTS, OPTIONAL_VALUE)
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(0, "output", "Result, has same type as input, with H and W dimensions reduced.", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          // Type inference
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          if (ONNX_NAMESPACE::hasNInputShapes(ctx, 1)) {
            const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            const auto input_rank = input_shape.dim_size();
            if (input_rank != 4) fail_shape_inference("Input's shape must be 4-D");

            // parse necessary attributes for futher processing
            std::vector<int64_t> border;
            bool border_present = getRepeatedAttribute(ctx, "border", border);
            if (!border_present || border.size() != 4)
              fail_shape_inference(
                  "'Border' attribute must be present and must contain exactly 4 values - "
                  "(left_border, top_border, right_border, bottom_border)");

            std::vector<int64_t> scale;
            bool scale_present = getRepeatedAttribute(ctx, "scale", scale);
            if (scale_present && scale.size() != 2)
              fail_shape_inference("'Scale' must contain exactly 2 values - (height, width)");

            // actual shape inference processing
            // [N, C] can be copied over from the input as is
            *output_shape->mutable_dim(static_cast<int>(0)) = input_shape.dim(static_cast<int>(0));
            *output_shape->mutable_dim(static_cast<int>(1)) = input_shape.dim(static_cast<int>(1));

            // process 'H' and 'W'
            if (!utils::HasDimValue(input_shape.dim(static_cast<int>(2))) ||
                !utils::HasDimValue(input_shape.dim(static_cast<int>(3)))) {
              // either height and width input has symbolic dims, so can't proceed further
              // add two dims as placeholders for output_H and output_W and return
              output_shape->add_dim();
              output_shape->add_dim();
              return;
            }

            int64_t H = input_shape.dim(static_cast<int>(2)).dim_value();
            int64_t W = input_shape.dim(static_cast<int>(3)).dim_value();

            int64_t left_border = border[0], top_border = border[1], right_border = border[2],
                    bottom_border = border[3];

            if (H < top_border + bottom_border)
              fail_shape_inference("Input's height (", H,
                                   ") needs to be greater than or equal to "
                                   "the top_border (",
                                   top_border, ") + bottom_border (", bottom_border, ")");

            if (W < left_border + right_border)
              fail_shape_inference("Input's width (", W,
                                   ") needs to be greater than or equal to "
                                   "the left_border (",
                                   left_border, ") + right_border (", right_border, ")");

            int64_t bottom_limit = H - bottom_border;
            int64_t right_limit = W - right_border;

            // scale = (height, width)
            if (!scale.empty()) {
              bottom_limit = top_border + scale[0];
              right_limit = left_border + scale[1];

              if (H < bottom_limit)
                fail_shape_inference("Input's height (", H, ") needs to be greater than or equal to the top_border (",
                                     top_border, ") + scale[0] (", scale[0], ")");

              if (W < right_limit)
                fail_shape_inference("Input's width (", W, ") needs to be greater than or equal to the left_border (",
                                     left_border, ") + scale[1] (", scale[1], ")");
            }

            auto* h_output_dim = output_shape->add_dim();
            h_output_dim->set_dim_value(bottom_limit - top_border);

            auto* w_output_dim = output_shape->add_dim();
            w_output_dim->set_dim_value(right_limit - left_border);
          } else {
            // Rank Inference at the very least
            // (We know that the output is going to be 4-D)
            for (int i = 0; i < 4; ++i) {
              output_shape->add_dim();
            }
          }
        }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    DynamicSlice, 10,
    OpSchema()
        .Deprecate()
        .SetDoc(DynamicSlice_ver1_doc)
        .Input(0, "data", "Tensor of data to extract slices from.", "T")
        .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
        .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes", "Tind")
        .Input(3, "axes", "1-D tensor of axes that `starts` and `ends` apply to.", "Tind", OpSchema::Optional)
        .Output(0, "output", "Sliced data tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types"));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ScaledTanh, 10,
    OpSchema()
        .Deprecate()
        .Attr("alpha", "Scaling value", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("beta", "Scaling value", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Input(0, "input", "Input tensor", "T")
        .Output(0, "output",
                "The scaled hyperbolic tangent values of the input tensor "
                "computed element-wise",
                "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

// End of ONNX exp ops(Affine, Crop, ParametricSoftplus, ImageScaler, ThresholdedRelu, DynamicSlice, ScaledTanh, MVN)
// old version history maintenance
}  // namespace contrib
}  // namespace onnxruntime
