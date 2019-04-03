// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/attn_lstm_schema_defs.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/range_schema_defs.h"
#include "core/graph/contrib_ops/reverse_sequence_schema_defs.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

#ifdef MICROSOFT_INTERNAL
#include "core/graph/contrib_ops/internal_schema_defs.h"
#endif

namespace ONNX_NAMESPACE {
void convPoolTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, bool use_dilation, bool require_kernel_shape);
}
namespace onnxruntime {
namespace contrib {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL;

void matmulShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int input1Idx, int input2Idx) {
  if (!hasInputShape(ctx, input1Idx) && !hasInputShape(ctx, input2Idx)) {
    return;
  }

  const auto shape0 = ctx.getInputType(input1Idx)->tensor_type().shape();
  const auto shape1 = ctx.getInputType(input2Idx)->tensor_type().shape();

  if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

  // First promote each shape to at least rank-2. This logic is
  // specific to matmul, not generic broadcasting.
  {
    if (shape0.dim_size() == 1) {
      shapeL.add_dim()->set_dim_value(1);
      *shapeL.add_dim() = shape0.dim(0);
    } else {
      *shapeL.mutable_dim() = shape0.dim();
    }
    if (shape1.dim_size() == 1) {
      *shapeR.add_dim() = shape1.dim(0);
      shapeR.add_dim()->set_dim_value(1);
    } else {
      *shapeR.mutable_dim() = shape1.dim();
    }
  }

  // Check for compatible matrix multiply dimensions
  {
    auto dimL = shapeL.dim(shapeL.dim_size() - 1);
    auto dimR = shapeR.dim(shapeR.dim_size() - 2);
    if (dimL.has_dim_value() && dimR.has_dim_value() &&
        dimL.dim_value() != dimR.dim_value()) {
      fail_shape_inference("Incompatible dimensions for matrix multiplication");
    }
  }

  ONNX_NAMESPACE::TensorShapeProto resultShape;

  // Now call out to generic multidimensional broadcasting for
  // the broadcastable prefixes.
  {
    ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
    for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
      *prefixShapeL.add_dim() = shapeL.dim(i);
    }
    for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
      *prefixShapeR.add_dim() = shapeR.dim(i);
    }
    bidirectionalBroadcastShapeInference(
        prefixShapeL, prefixShapeR, resultShape);
  }

  // Back to matmul-specific. Add the trailing dimensions back in.
  {
    if (shape0.dim_size() != 1) {
      *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
    }
    if (shape1.dim_size() != 1) {
      *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
    }
  }

  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = resultShape;
}

void convPoolShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    bool use_dilation,
    bool require_kernel_shape,
    int input1Idx, int input2Idx) {
  if (!hasInputShape(ctx, input1Idx)) {
    return;
  }

  // if kernel shape is an input (and not attribute)
  // we need the shape of the second input.
  if (!require_kernel_shape && !hasNInputShapes(ctx, input2Idx)) {
    return;
  }

  // don't bother with legacy auto_pad for now
  if (ctx.getAttribute("auto_pad")) {
    return;
  }

  auto input_shape = ctx.getInputType(input1Idx)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor must have atleast 2 dimensions");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // Pooling operations don't support dilation, only Conv. For
  // simplicity of the code, we just treat them as having all-1s
  // dilation.
  std::vector<int64_t> dilations;
  if (use_dilation && getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      fail_shape_inference("Attribute dilations has incorrect size");
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  int64_t groups = getAttribute(ctx, "group", 1);
  if (groups != 1) {
    return;  // we don't handle the group case.
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size");
    }
  } else if (require_kernel_shape) {
    fail_shape_inference("Attribute kernel_shape must be specified");
  } else {
    auto second_input_shape = ctx.getInputType(input2Idx)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
  }

  auto output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  if (require_kernel_shape) {
    // add the first two dimensions from the input.
    *output_shape->add_dim() = input_shape.dim(0);
    *output_shape->add_dim() = input_shape.dim(1);
  } else {
    *output_shape->add_dim() = input_shape.dim(0);
    auto& second_input_shape = getInputShape(ctx, 1);
    if (second_input_shape.dim_size() < 1) {
      fail_shape_inference("Second input tensor has wrong dimension");
    }
    *output_shape->add_dim() = second_input_shape.dim(0);
  }

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto newdim = output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }
    // how big is the input, including padding
    int64_t effective_input_size = input_shape.dim(2 + i).dim_value();
    effective_input_size += pads[i];
    effective_input_size += pads[i + kernel_shape_size];

    int64_t effective_kernel_size = kernel_shape[i];
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_size = (effective_kernel_size - 1) * dilations[i] + 1;

    // how many times we can move the kernel from it's initial position, based
    // on the stride
    int64_t strided_kernel_positions =
        (effective_input_size - effective_kernel_size) / strides[i];

    // add in the initial position
    newdim->set_dim_value(1 + strided_kernel_positions);
  }

  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto second_output_shape =
        ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    second_output_shape->CopyFrom(*output_shape);
  }
}

void RegisterContribSchemas() {
  // ONNX exp ops(Affine, Crop, ParametricSoftplus, ImageScaler) old version history maintainance
  static const char* Affine_ver1_doc = R"DOC(
Affine takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the affine function, y = alpha * x + beta,
is applied to the tensor elementwise.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Affine)
      .SinceVersion(1)
      .SetDoc(Affine_ver1_doc)
      .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, 1.0f)
      .Attr("beta", "Value of beta", AttributeProto::FLOAT, 0.0f)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D output tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* ParametricSoftplus_ver1_doc = R"DOC(
ParametricSoftplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = alpha * ln(exp(beta * x) + 1), is applied to
the tensor elementwise.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(ParametricSoftplus)
      .SinceVersion(1)
      .SetDoc(ParametricSoftplus_ver1_doc)
      .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, OPTIONAL)
      .Attr("beta", "Value of beta", AttributeProto::FLOAT, OPTIONAL)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D input tensor", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* ImageScaler_ver1_doc =
      R"DOC(Scale and bias the input image. Bias values are stored in
the same ordering as the image pixel format.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(ImageScaler)
      .SinceVersion(1)
      .SetDoc(ImageScaler_ver1_doc)
      .Attr("bias", "Bias applied to each channel, same size as C.", AttributeProto::FLOATS, OPTIONAL)
      .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same shape and type as input", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* Crop_ver1_doc =
      R"DOC(Crop and image to the specified spatial dimensions. If scale is given,
then optionally start the crop offset by the left/top border amounts.
If scale is not provided, crop the borders as provided.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Crop)
      .SinceVersion(1)
      .SetDoc(Crop_ver1_doc)
      .Attr("border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).", AttributeProto::INTS, OPTIONAL)
      .Attr("scale", "A 1-D values of (height, width).", AttributeProto::INTS, OPTIONAL)
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same type as input, with H and W dimensions reduced.", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.");

  static const char* ThresholdedRelu_ver1_doc = R"DOC( 
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data 
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise, 
is applied to the tensor elementwise. )DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(ThresholdedRelu)
      .SinceVersion(1)
      .SetDoc(ThresholdedRelu_ver1_doc)
      .Attr("alpha", "Threshold value", AttributeProto::FLOAT, 1.0f)
      .Input(0, "X", "Input tensor", "T")
      .Output(0, "Y", "Output tensor", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput); 

  static const char* DynamicSlice_ver1_doc = R"DOC(
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

  ONNX_CONTRIB_OPERATOR_SCHEMA(DynamicSlice)
      .SinceVersion(1)
      .SetDoc(DynamicSlice_ver1_doc)
      .Input(0, "data", "Tensor of data to extract slices from.", "T")
      .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
      .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes", "Tind")
      .Input(3, "axes", "1-D tensor of axes that `starts` and `ends` apply to.", "Tind", OpSchema::Optional)
      .Output(0, "output", "Sliced data tensor.", "T")
      .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types");

  ONNX_CONTRIB_OPERATOR_SCHEMA(Affine)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(Affine_ver1_doc)
      .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, 1.0f)
      .Attr("beta", "Value of beta", AttributeProto::FLOAT, 0.0f)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D output tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(ParametricSoftplus)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(ParametricSoftplus_ver1_doc)
      .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, OPTIONAL)
      .Attr("beta", "Value of beta", AttributeProto::FLOAT, OPTIONAL)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D input tensor", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(ImageScaler)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(ImageScaler_ver1_doc)
      .Attr("bias", "Bias applied to each channel, same size as C.", AttributeProto::FLOATS, OPTIONAL)
      .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same shape and type as input", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(Crop)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(Crop_ver1_doc)
      .Attr("border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).", AttributeProto::INTS, OPTIONAL)
      .Attr("scale", "A 1-D values of (height, width).", AttributeProto::INTS, OPTIONAL)
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same type as input, with H and W dimensions reduced.", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(DynamicSlice)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(DynamicSlice_ver1_doc)
      .Input(0, "data", "Tensor of data to extract slices from.", "T")
      .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
      .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes", "Tind")
      .Input(3, "axes", "1-D tensor of axes that `starts` and `ends` apply to.", "Tind", OpSchema::Optional)
      .Output(0, "output", "Sliced data tensor.", "T")
      .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types");

  // End of ONNX exp ops(Affine, Crop, ParametricSoftplus, ImageScaler) old version history maintainance

  ONNX_CONTRIB_OPERATOR_SCHEMA(SampleOp)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "input", "T")
      .Output(0, "Y", "output", "T")
      .TypeConstraint(
          "T",
          ONNX_NAMESPACE::OpSchema::numeric_types_for_math_reduction(),
          "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
      .SetDoc(R"DOC(
Sample echo operator.)DOC");

  // register schemas for more operators here
  ONNX_CONTRIB_OPERATOR_SCHEMA(MaxpoolWithMask)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(For internal use.)DOC")
      .Attr(
          "auto_pad",
          "",
          AttributeProto::STRING,
          std::string("NOTSET"))
      .Attr(
          "kernel_shape",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr("pads",
            "",
            AttributeProto::INTS, OPTIONAL)
      .Attr(
          "storage_order",
          "",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "strides", "", AttributeProto::INTS, OPTIONAL)
      .Input(
          0,
          "X",
          "",
          "T")
      .Input(1, "M", "mask", "tensor(int32)")
      .Output(
          0,
          "Y",
          "",
          "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input0 and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::convPoolTypeAndShapeInference(ctx, false, true);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(FusedConv)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
The fused convolution operator schema is the same as Conv besides it includes an attribute
activation.)DOC")
      .Attr(
          "auto_pad",
          "",
          AttributeProto::STRING,
          std::string("NOTSET"))
      .Attr(
          "kernel_shape",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "dilations",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "strides", "", AttributeProto::INTS, OPTIONAL)
      .Attr("pads",
            "",
            AttributeProto::INTS, OPTIONAL)
      .Attr(
          "group",
          "",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Attr(
          "activation",
          "",
          AttributeProto::STRING,
          OPTIONAL)
      .Attr(
          "alpha",
          "",
          AttributeProto::FLOAT,
          OPTIONAL)
      .Input(
          0,
          "X",
          "",
          "T")
      .Input(
          1,
          "W",
          "",
          "T")
      .Input(2, "B", "", "T", OpSchema::Optional)
      .Output(
          0,
          "Y",
          "",
          "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::convPoolTypeAndShapeInference(ctx, true, false);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(FusedGemm)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
The FusedGemm operator schema is the same as Gemm besides it includes attributes
activation and leaky_relu_alpha.)DOC")
      .Input(
          0,
          "A",
          "Input tensor A. "
          "The shape of A should be (M, K) if transA is 0, "
          "or (K, M) if transA is non-zero.",
          "T")
      .Input(
          1,
          "B",
          "Input tensor B. "
          "The shape of B should be (K, N) if transB is 0, "
          "or (N, K) if transB is non-zero.",
          "T")
      .Input(
          2,
          "C",
          "Input tensor C. "
          "The shape of C should be unidirectional broadcastable to (M, N).",
          "T")
      .Output(0, "Y", "Output tensor of shape (M, N).", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)",
           "tensor(float)",
           "tensor(double)",
           "tensor(uint32)",
           "tensor(uint64)",
           "tensor(int32)",
           "tensor(int64)"},
          "Constrain input and output types to float/int tensors.")
      .Attr(
          "transA",
          "Whether A should be transposed",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "transB",
          "Whether B should be transposed",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "alpha",
          "Scalar multiplier for the product of input tensors A * B.",
          AttributeProto::FLOAT,
          1.0f)
      .Attr(
          "beta",
          "Scalar multiplier for input tensor C.",
          AttributeProto::FLOAT,
          1.0f)
      .Attr(
          "activation",
          "",
          AttributeProto::STRING,
          OPTIONAL)
      .Attr(
          "leaky_relu_alpha",
          "",
          AttributeProto::FLOAT,
          OPTIONAL)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (hasNInputShapes(ctx, 2)) {
          auto transAAttr = ctx.getAttribute("transA");
          bool transA =
              transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
          auto transBAttr = ctx.getAttribute("transB");
          bool transB =
              transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
          auto& first_input_shape = getInputShape(ctx, 0);
          auto& second_input_shape = getInputShape(ctx, 1);
          if (first_input_shape.dim_size() != 2)
            fail_shape_inference("First input does not have rank 2");
          if (second_input_shape.dim_size() != 2)
            fail_shape_inference("Second input does not have rank 2");
          updateOutputShape(
              ctx,
              0,
              {first_input_shape.dim(transA ? 1 : 0),
               second_input_shape.dim(transB ? 0 : 1)});
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ExpandDims)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "input", "T")
      .Input(1, "axis", "Specified axis to insert a dimension", "tensor(int32)")
      .Output(0, "Y", "output", "T")
      .TypeConstraint(
          "T",
          ONNX_NAMESPACE::OpSchema::all_tensor_types(),
          "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // Shape inference
        if (!hasInputShape(ctx, 0))
          return;

        auto& input_shape = getInputShape(ctx, 0);
        const int rank = input_shape.dim_size();
        const ONNX_NAMESPACE::TensorProto* axis_initializer = ctx.getInputData(1);
        if (!axis_initializer)
          return;
        const int axis = axis_initializer->int32_data()[0];
        if (axis > rank || axis < -rank - 1) {
          fail_shape_inference("Input axis is invalid: ", axis);
        }
        int pos = axis >= 0 ? axis : rank + axis - 1;
        ONNX_NAMESPACE::TensorShapeProto output_shape;
        for (int i = 0; i < pos; ++i) {
          output_shape.add_dim();
          *(output_shape.mutable_dim(i)) = input_shape.dim(i);
        }
        output_shape.add_dim();
        output_shape.mutable_dim(pos)->set_dim_value(1);
        for (int i = pos + 1; i < rank + 1; ++i) {
          output_shape.add_dim();
          *(output_shape.mutable_dim(i)) = input_shape.dim(i - 1);
        }
        updateOutputShape(ctx, 0, output_shape);
      })
      .SetDoc(R"DOC(ExpandDims echo operator.)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(AttnLSTM, RegisterAttnLSTMContribOpSchema);
  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(Range, RegisterRangeOpSchema);
  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(ReverseSequence, RegisterReverseSequenceOpSchema);

  static const char* Tokenizer_ver1_doc = R"DOC(
  Tokenizer divides each string in X into a vector of strings along the last axis. Allowed input shapes are [C] and [N, C].
  If the maximum number of tokens found per input string is D, the output shape would be [N, C, D] when input shape is [N, C].
  Similarly, if input shape is [C] then the output should be [C, D]. Tokenizer has two different operation modes.
  The first mode is selected when "tokenexp" is not set and "separators" is set. If "tokenexp" is set and "separators" is not set,
  the second mode will be used. The first mode breaks each input string into tokens by removing separators.

  Let's assume "separators" is [" "] and consider an example.
  If input is

  ["Hello World", "I love computer science !"] whose shape is [2],

  then the output would be

 [["Hello", "World", padvalue, padvalue, padvalue],
 ["I", "love", "computer", "science", "!"]]

 whose shape is [2, 5] because you can find at most 5 tokens per input string.
 Note that the input at most can have two axes, so 3-D and higher dimension are not supported.

 For each input string, the second mode searches matches of "tokenexp" and each match will be a token in Y.
 The matching of "tokenexp" is conducted greedily (i.e., a match should be as long as possible).
 This operator searches for the first match starting from the beginning of the considered string,
 and then launches another search starting from the first remained character after the first matched token.
 If no match found, this operator will remove the first character from the remained string and do another search.
 This procedure will be repeated until reaching the end of the considered string.

  Let's consider another example to illustrate the effect of setting "mark" to true.
  If input is ["Hello", "World"],
  then the corresponding output would be [0x02, "Hello", "World", 0x03].
  This implies that if mark is true, [C]/[N, C] - input's output shape becomes [C, D+2]/[N, C, D+2].
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Tokenizer)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "Strings to tokenize", "T")
      .Output(0, "Y", "Tokenized strings", "T")
      .TypeConstraint(
          "T",
          {"tensor(string)"},
          "Input/Output is a string tensor")
      .Attr(
          "mark",
          "Boolean whether to mark the beginning/end character with start of text character (0x02)/end of text character (0x03).",
          AttributeProto::INT)
      .Attr(
          "pad_value",
          "The string used to pad output tensors when the tokens extracted doesn't match the maximum number of tokens found. If start/end markers are needed, padding will appear outside the markers.",
          AttributeProto::STRING)
      .Attr(
          "tokenexp",
          "An optional string. Token's regular expression in basic POSIX format"
          " (http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap09.html#tag_09_03)."
          " If set, tokenizer may produce tokens matching the specified pattern. Note that one and only of"
          " 'tokenexp' and 'separators' should be set.",
          AttributeProto::STRING,
          OPTIONAL)
      .Attr(
          "separators",
          "an optional list of strings (type: AttributeProto::STRINGS), each single string in this attribute is a separator."
          " Two consecutive segments in X connected by a separator would be divided into two tokens."
          " For example, if the input is \"Hello World!\" and this attribute contains only one space character,"
          " the corresponding output would be [\"Hello\", \"World!\"]. To achieve character-level tokenization,"
          " one should set the separators to [\"\"], which contains only one empty string."
          " If 'separators' is a L-element array, there will be L rounds of tokenization using one stop word."
          " More specifically, in the first round, the first element in 'separators' is used to tokenize each string in the input."
          " Then, the second element in 'separators' will be used to tokenize the resulted strings produced at the first round.",
          AttributeProto::STRINGS,
          OPTIONAL)
      .Attr(
          "mincharnum",
          "Minimum number of characters allowed in the output. For example, if mincharnum is 2, tokens such as \"A\" and \"B\" would be ignored",
          AttributeProto::INT)
      .SetDoc(Tokenizer_ver1_doc)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // Shape inference
        if (!hasInputShape(ctx, 0))
          return;

        ONNX_NAMESPACE::TensorShapeProto output_shape;
        auto& input_shape = getInputShape(ctx, 0);
        auto& dims = input_shape.dim();
        if (dims.size() < 1 || dims.size() > 2) {
          fail_shape_inference("Input dimensions are either [C] or [N][C] allowed");
        }
        for (auto& dim : dims) {
          *output_shape.add_dim() = dim;
        }
        // Add the last unknown dimension
        output_shape.add_dim();
        updateOutputShape(ctx, 0, output_shape);
      });

  // Operators for linear 8 bit quanitzation support.
  ONNX_CONTRIB_OPERATOR_SCHEMA(QuantizeLinear)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("axis", "The axis along which same quantization parameters are applied. It's optional. If it's not specified, it means per-tensor quantization and input 'x_scale' and 'x_zero_point' must be scalars. If it's specified, it means per 'axis' quantization and input 'x_scale' and 'x_zero_point' must be 1-D tensors.", AttributeProto::INT, false)
      .Input(0, "x", "N-D full precision Input tensor to be quantized.", "T1")
      .Input(1, "y_scale", "Scale for doing quantization to get 'y'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-axis quantization. If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'.", "T1")
      .Input(2, "y_zero_point", "Zero point for doing quantization to get 'y'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-axis quantization. If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'.", "T2")
      .Output(0, "y", "N-D quantized output tensor. It has same shape as input 'x'.", "T2")
      .TypeConstraint(
          "T1",
          {"tensor(float)"},
          "Constrain 'x', 'y_scale' to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain 'y_zero_point' and 'y' to 8-bit integer tensors.")
      .SetDoc(R"DOC(
The linear quantization operator. It consumes a full precision data, a scale, a zero point and computes the quantized data.
The quantization formula is y = (x / y_scale) + y_zero_point. For (x / y_scale), it computes the nearest integer value to arg (in floating-point format),
 rounding halfway cases away from zero. Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per 'axis').)DOC")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 2, 0);

        if (!hasInputShape(ctx, 0))
          return;

        auto& input_shape = getInputShape(ctx, 0);
        updateOutputShape(ctx, 0, input_shape);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(DequantizeLinear)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("axis", "the axis along which same quantization parameters are applied. It's optional. If it's not specified, it means per-tensor quantization and input 'x_scale' and 'x_zero_point' must be scalars. If it's specified, it means per 'axis' quantization and input 'x_scale' and 'x_zero_point' must be 1-D tensors.", AttributeProto::INT, false)
      .Input(0, "x", "N-D quantized Input tensor to be de-quantized.", "T2")
      .Input(1, "x_scale", "Scale for input 'x'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-axis quantization. If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'.", "T1")
      .Input(2, "x_zero_point", "Zero point for input 'x'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-axis quantization. If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'.", "T2")
      .Output(0, "y", "N-D full precision output tensor. It has same shape as input 'x'.", "T1")
      .TypeConstraint(
          "T1",
          {"tensor(float)"},
          "Constrain 'y', 'x_scale' to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain 'x_zero_point' and 'x' to 8-bit integer tensors.")
      .SetDoc(R"DOC(
The linear de-quantization operator. It consumes a quantized data, a scale, a zero point and computes the full precision data.
The dequantization formula is y = (x - x_zero_point) * x_scale.
Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per 'axis').)DOC")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto y_type = ctx.getOutputType(0);
        // only float is supported
        y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);

        if (!hasInputShape(ctx, 0))
          return;

        auto& input_shape = getInputShape(ctx, 0);
        updateOutputShape(ctx, 0, input_shape);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearMatMul)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
It consumes two quantized input tensors, their scales and zero points, and output's scale and zero point, and computes
the quantized output. The quantization formula is x_quantized = (x_fp32 / x_scale) + x_zero_point. For (x_fp32 / x_scale),
it computes the nearest integer value to arg (in floating-point format), rounding halfway cases away from zero.
Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per row for a and per column for b).
If scale and zero point are 1D tensor, the number of elements of scale and zero point tensor of input 'a' and output 'y'
should be equal to the number of rows of input 'a', and the number of elements of scale and zero point tensor of input 'b'
should be equal to the number of columns of input 'b'. The production MUST never overflow. The accumulation may overflow in 32 bits
if the input is 8 bits or in 64 bits if the input is 16 bits.)DOC")
      .Input(0, "a", "N-dimensional quantized matrix a", "T1")
      .Input(1, "a_scale", "scale of quantized input a", "tensor(float)")
      .Input(2, "a_zero_point", "zero point of quantized input a", "T1")
      .Input(3, "b", "N-dimensional quantized matrix b", "T2")
      .Input(4, "b_scale", "scale of quantized input b", "tensor(float)")
      .Input(5, "b_zero_point", "zero point of quantized input b", "T2")
      .Input(6, "y_scale", "scale of quantized output y", "tensor(float)")
      .Input(7, "y_zero_point", "zero point of quantized output y", "T3")
      .Output(0, "y", "Quantized matrix multiply results from a * b", "T3")
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)", "tensor(int16)", "tensor(uint16)"}, "Constrain input a and its zero point data types as 8-bit or 16-bit integer tensor")
      .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)", "tensor(int16)", "tensor(uint16)"}, "Constrain input b and its zero point data types as 8-bit or 16-bit integer tensor")
      .TypeConstraint("T3", {"tensor(int8)", "tensor(uint8)", "tensor(int16)", "tensor(uint16)"}, "Constrain output y and its zero point data types as 8-bit or 16-bit integer tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto a_type = ctx.getInputType(0);
        auto b_type = ctx.getInputType(3);
        auto y_type = ctx.getOutputType(0);
        if (nullptr == a_type || nullptr == b_type || nullptr == y_type ||
            a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
            b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
          fail_type_inference(
              "inputs are expected to have tensor type and output type should not be null.");
        }

        if (ONNX_NAMESPACE::TensorProto::UINT8 == a_type->tensor_type().elem_type() &&
            ONNX_NAMESPACE::TensorProto::UINT8 == b_type->tensor_type().elem_type()) {
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::UINT8);
        } else {
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT8);
        }

        matmulShapeInference(ctx, 0, 3);
      });

  const char* auto_pad_doc =
      "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
      "default value is NOTSET, which means explicit padding is used. "
      "SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input."
      "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
      "beginning for SAME_LOWER. VALID mean no padding.";

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearConv)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
The convolution operator consumes a quantized input tensor, its scale and zero point,
a quantized filter, its scale and zero point, and output's scale and zero point,
and computes the quantized output. Each scale and zero point pair must have same shape.
It means they must be either scalars (per tensor) or 1-D tensors (per channel).
The production MUST never overflow. The accumulation may overflow in 32 bits
if the input is 8 bits or in 64 bits if the input is 16 bits.)DOC")
      .Input(
          0,
          "x",
          "Input data tensor from previous layer; "
          "has size (N x C x H x W), where N is the batch size, "
          "C is the number of channels, and H and W are the "
          "height and width. Note that this is for the 2D image. "
          "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
          "Optionally, if dimension denotation is "
          "in effect, the operation expects input data tensor "
          "to arrive with the dimension denotation of [DATA_BATCH, "
          "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
          "T1")
      .Input(1, "x_scale", "Scale tensor for input 'x'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of channels of input 'x'.", "tensor(float)")
      .Input(2, "x_zero_point", "Zero point tensor for input 'x'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of channels of input 'x'.", "T1")
      .Input(
          3,
          "w",
          "The weight tensor that will be used in the "
          "convolutions; has size (M x C/group x kH x kW), where C "
          "is the number of channels, and kH and kW are the "
          "height and width of the kernel, and M is the number "
          "of feature maps. For more than 2 dimensions, the "
          "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
          "where (k1 x k2 x ... kn) is the dimension of the kernel. "
          "Optionally, if dimension denotation is in effect, "
          "the operation expects the weight tensor to arrive "
          "with the dimension denotation of [FILTER_OUT_CHANNEL, "
          "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
          "X.shape[1] == (W.shape[1] * group) == C "
          "(assuming zero based indices for the shape array). "
          "Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. ",
          "T2")
      .Input(4, "w_scale", "Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of channels of input 'w'.", "tensor(float)")
      .Input(5, "w_zero_point", "Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of channels of input 'w'.", "T2")
      .Input(6, "y_scale", "Scale tensor for output 'y'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of channels of input 'y'.", "tensor(float)")
      .Input(7, "y_zero_point", "Scale tensor for output 'y'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of channels of input 'y'.", "T3")
      .Input(8, "B", "Optional 1D bias to be added to the convolution, has size of M.", "T4", OpSchema::Optional)
      .Output(
          0,
          "y",
          "Output data tensor that contains the result of the "
          "convolution. The output dimensions are functions "
          "of the kernel size, stride size, and pad lengths.",
          "T3")
      .TypeConstraint(
          "T1",
          {"tensor(int8)", "tensor(uint8)", "tensor(int16)", "tensor(uint16)"},
          "Constrain input types to 8-bit or 16-bit integer tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(int8)", "tensor(uint8)", "tensor(int16)", "tensor(uint16)"},
          "Constrain filter types to 8-bit or 16-bit integer tensors.")
      .TypeConstraint(
          "T3",
          {"tensor(int8)", "tensor(uint8)", "tensor(int16)", "tensor(uint16)"},
          "Constrain output types to 8-bit or 16-bit integer tensors.")
      .TypeConstraint("T4", {"tensor(int32)", "tensor(uint32)"}, "Constrain bias type to 32-bit integer tensor.")
      .Attr(
          "auto_pad",
          auto_pad_doc,
          AttributeProto::STRING,
          std::string("NOTSET"))
      .Attr(
          "kernel_shape",
          "The shape of the convolution kernel. If not present, should be inferred from input 'w'.",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "dilations",
          "dilation value along each axis of the filter. If not present, the dilation defaults to 1 along each axis.",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "strides", "Stride along each axis. If not present, the stride defaults to 1 along each axis.", AttributeProto::INTS, OPTIONAL)
      .Attr("pads",
            "Padding for the beginning and ending along each axis, it can take any value greater than or equal to 0."
            "The value represent the number of pixels added to the beginning and end part of the corresponding axis."
            "`pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of"
            "pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`."
            "This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults"
            "to 0 along start and end of each axis.",
            AttributeProto::INTS, OPTIONAL)
      .Attr(
          "group",
          "number of groups input channels and output channels are divided into. default is 1.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto x_type = ctx.getInputType(0);
        auto w_type = ctx.getInputType(3);
        auto y_type = ctx.getOutputType(0);
        if (nullptr == x_type || nullptr == w_type || nullptr == y_type ||
            x_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
            w_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
          fail_type_inference(
              "inputs are expected to have tensor type and output type should not be null.");
        }

        if (ONNX_NAMESPACE::TensorProto::UINT8 == x_type->tensor_type().elem_type() &&
            ONNX_NAMESPACE::TensorProto::UINT8 == w_type->tensor_type().elem_type()) {
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::UINT8);
        } else {
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT8);
        }

        convPoolShapeInference(ctx, true, false, 0, 3);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ConvInteger)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
The integer convolution operator consumes an input tensor, a filter, and a padding value,
 and computes the output. The production MUST never overflow. The accumulation may overflow
 if and only if in 32 bits.)DOC")
      .Input(
          0,
          "x",
          "Input data tensor from previous layer; "
          "has size (N x C x H x W), where N is the batch size, "
          "C is the number of channels, and H and W are the "
          "height and width. Note that this is for the 2D image. "
          "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
          "Optionally, if dimension denotation is "
          "in effect, the operation expects input data tensor "
          "to arrive with the dimension denotation of [DATA_BATCH, "
          "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
          "T1")
      .Input(
          1,
          "w",
          "The weight tensor that will be used in the "
          "convolutions; has size (M x C/group x kH x kW), where C "
          "is the number of channels, and kH and kW are the "
          "height and width of the kernel, and M is the number "
          "of feature maps. For more than 2 dimensions, the "
          "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
          "where (k1 x k2 x ... kn) is the dimension of the kernel. "
          "Optionally, if dimension denotation is in effect, "
          "the operation expects the weight tensor to arrive "
          "with the dimension denotation of [FILTER_OUT_CHANNEL, "
          "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
          "X.shape[1] == (W.shape[1] * group) == C "
          "(assuming zero based indices for the shape array). "
          "Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. ",
          "T2")
      .Input(2, "x_zero_point",
             "Zero point tensor for input 'x'. It's optional and default value is 0. It could be a scalar or a 1-D tensor, "
             "which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number of elements "
             "should be equal to the number of channels of input 'x'.",
             "T1", OpSchema::Optional)
      .Input(3, "w_zero_point",
             "Scale tensor for input 'w'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
             "which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number "
             "of elements should be equal to the number of channels of input 'w'.",
             "T2", OpSchema::Optional)
      .Output(
          0,
          "y",
          "Output data tensor that contains the result of the "
          "convolution. The output dimensions are functions "
          "of the kernel size, stride size, and pad lengths.",
          "T3")
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input X and Z data types as 8-bit integer tensors")
      .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain input W data types as 8-bit integer tensor")
      .TypeConstraint("T3",
                      {"tensor(int32)", "tensor(uint32)"},
                      "Constrain output Y data types as 32-bits integer tensors."
                      "T3 must be tensor(uint32) when both T1 and T2 are tensor(uint8),"
                      "or must be tensor(int32) when either T1 or T2 is tensor(int8).")
      .Attr(
          "auto_pad",
          auto_pad_doc,
          AttributeProto::STRING,
          std::string("NOTSET"))
      .Attr(
          "kernel_shape",
          "The shape of the convolution kernel. If not present, should be inferred from input 'w'.",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "dilations",
          "dilation value along each axis of the filter. If not present, the dilation defaults to 1 along each axis.",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "strides", "Stride along each axis. If not present, the stride defaults to 1 along each axis.", AttributeProto::INTS, OPTIONAL)
      .Attr("pads",
            "Padding for the beginning and ending along each axis, it can take any value greater than or equal to 0."
            "The value represent the number of pixels added to the beginning and end part of the corresponding axis."
            "`pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of"
            "pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`."
            "This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults"
            "to 0 along start and end of each axis.",
            AttributeProto::INTS, OPTIONAL)
      .Attr(
          "group",
          "number of groups input channels and output channels are divided into. default is 1.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto x_type = ctx.getInputType(0);
        auto w_type = ctx.getInputType(1);
        auto y_type = ctx.getOutputType(0);
        if (nullptr == x_type || nullptr == w_type || nullptr == y_type ||
            x_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
            w_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
          fail_type_inference(
              "inputs are expected to have tensor type and output type should not be null.");
        }

        // Right now we only support int32
        y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT32);

        convPoolShapeInference(ctx, true, false, 0, 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MatMulInteger)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
 The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.)DOC")
      .Input(0, "A", "N-dimensional matrix A", "T1")
      .Input(1, "B", "N-dimensional matrix B", "T2")
      .Input(2, "a_zero_point",
             "Zero point tensor for input 'A'. It's optional and default value is 0. It could be a scalar or a 1-D tensor, "
             "which means a per-tensor or per-row quantization. If it's a 1-D tensor, its number of elements "
             "should be equal to the number of rows of input 'A'.",
             "T1", OpSchema::Optional)
      .Input(3, "b_zero_point",
             "Scale tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
             "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
             "of elements should be equal to the number of columns of input 'B'.",
             "T2", OpSchema::Optional)

      .Output(0, "Y", "Matrix multiply results from A * B", "T3")
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input A data types as 8-bit integer tensor")
      .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain input B data types as 8-bit integer tensor")
      .TypeConstraint("T3",
                      {"tensor(int32)", "tensor(uint32)"},
                      "Constrain output Y data types as 32-bit integer tensor."
                      "T3 must be tensor(uint32) when both T1 and T2 are tensor(uint8),"
                      "or must be tensor(int32) when either T1 or T2 is tensor(int8).")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto a_type = ctx.getInputType(0);
        auto b_type = ctx.getInputType(1);
        auto y_type = ctx.getOutputType(0);
        if (nullptr == a_type || nullptr == b_type || nullptr == y_type ||
            a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
            b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
          fail_type_inference(
              "inputs are expected to have tensor type and output type should not be null.");
        }

        // Right now we only support int32
        y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT32);

        matmulShapeInference(ctx, 0, 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReduceSumInteger)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
Computes the sum of the low-precision input tensor's element along the provided axes.
The resulting tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0,
then the resulting tensor have the reduced dimension pruned. The above behavior is similar to numpy,
with the exception that numpy default keepdims to False instead of True.)DOC")
      .Input(0, "data", "An input tensor.", "T1")
      .Output(0, "reduced", "Reduced output tensor.", "T2")
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input type to 8-bit integer tensor.")
      .TypeConstraint("T2",
                      {"tensor(int32)", "tensor(uint32)"},
                      "Constrain output data type to 32-bit integer tensor."
                      "T2 must be tensor(uint32) when T1 is tensor(uint8),"
                      "or must be tensor(int32) when T1 is tensor(int8).")
      .Attr(
          "axes",
          "A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor.",
          AttributeProto::INTS)
      .Attr(
          "keepdims",
          "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
          AttributeProto::INT);

  ONNX_CONTRIB_OPERATOR_SCHEMA(NonMaxSuppression)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
Note: The boxes doesn't has class dimension which means it alwasy has scores calculated for different classes on same box.)DOC")
      .Input(
          0,
          "boxes",
          "An input tensor with shape [num_batches, spatial_dimension, 4]. The single box data format is indicated by center_point_box.",
          "tensor(float)")
      .Input(
          1,
          "scores",
          "An input tensor with shape [num_batches, num_classes, spatial_dimension]",
          "tensor(float)")
      .Input(
          2,
          "max_output_boxes_per_class",
          "Integer representing the maximum number of boxes to be selected per batch per class. It is a scalar. Value should be greater than 0",
          "tensor(int32)",
          OpSchema::Optional)
      .Input(
          3,
          "iou_threshold",
          "Float representing the threshold for deciding whether boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1].",
          "tensor(float)",
          OpSchema::Optional)
      .Input(
          4,
          "score_threshold",
          "Float representing the threshold for deciding when to remove boxes based on score. It is a scalar",
          "tensor(float)",
          OpSchema::Optional)
      .Output(
          0,
          "selected_indices",
          "selected indices from the boxes tensor. [num_selected_indices, 3], the selected indices format is [batch_index, class_index, box_index].",
          "tensor(int32)")
      .Attr(
          "center_point_box",
          "Integer indicate the format of the box data. The default is 0."
          "0 - the box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box corners"
          "and the coordinates can be provided as normalized (i.e., lying in the interval [0, 1]) or absolute. Mostly used for TF models."
          "1 - the box data is supplied as [x_center, y_center, width, height]. Mostly used for Pytoch models.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto selected_indices_type = ctx.getOutputType(0)->mutable_tensor_type();
        selected_indices_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MurmurHash3)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(The underlying implementation is MurmurHash3_x86_32 generating low latency 32bits hash suitable for implementing lookup tables, Bloom filters, count min sketch or feature hashing.)DOC")
      .Input(0, "X", "An input tensor to hash.", "T1")
      .Output(0, "Y", "32-bit hash value.", "T2")
      .TypeConstraint("T1", {"tensor(uint32)", "tensor(int32)", "tensor(string)"}, "Constrain input type to unsigned or signed 32-bit integer tensor, or string tensor. It should be utf-8 encoded if using unicode.")
      .TypeConstraint("T2", {"tensor(uint32)", "tensor(int32)"}, "Constrain output type to unsigned and signed 32-bit integer tensor.")
      .Attr(
          "seed",
          "Seed for the hashing algorithm, unsigned 32-bit integer, default to 0.",
          AttributeProto::INT,
          (int64_t)0LL)
      .Attr(
          "positive",
          "If value is 1, output type is uint32_t, else int32_t. Default value is 1.",
          AttributeProto::INT,
          (int64_t)1LL)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // type inference
        auto positive_attr = ctx.getAttribute("positive");
        bool is_positive =
            positive_attr ? (static_cast<int>(positive_attr->i()) == 1 ? true : false) : true /* default value if attribute not present */;
        auto output_data_type = ctx.getOutputType(0)->mutable_tensor_type();
        if (is_positive) {
          output_data_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32);
        } else {
          output_data_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32);
        }

        // Shape inference
        if (!hasInputShape(ctx, 0))
          return;

        auto& input_shape = getInputShape(ctx, 0);
        updateOutputShape(ctx, 0, input_shape);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GatherND)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "data", "Tensor of rank r >= 1.", "T")
      .Input(1, "indices", "Tensor of rank q >= 1.", "Tind")
      .Output(0, "output", "Tensor of rank q-1+r-indices[-1].", "T")
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types(),
          "Constrain input and output types to any tensor type.")
      .TypeConstraint(
          "Tind",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain indice type to int32 or int64")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (!hasNInputShapes(ctx, 2)) {
          fail_shape_inference("GatherND requires two tensor inputs.");
        }
        auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
        auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
        auto data_rank = data_shape.dim_size();
        auto indices_rank = indices_shape.dim_size();
        if (data_rank < 1 || indices_rank < 1) {
          fail_shape_inference("both data and indices tensor need to have rank larger than zero.");
        }
        auto last_indice_dimension = indices_shape.dim(indices_rank - 1).dim_value();
        if (last_indice_dimension > data_rank) {
          fail_shape_inference("last dimension of indices must not be larger and rank of data tensor");
        }
        for (int i = 0; i < indices_rank - 1; ++i) {
          *ctx.getOutputType(0)
               ->mutable_tensor_type()
               ->mutable_shape()
               ->add_dim() = indices_shape.dim(i);
        }
        for (int i = static_cast<int>(last_indice_dimension); i < data_rank; ++i) {
          *ctx.getOutputType(0)
               ->mutable_tensor_type()
               ->mutable_shape()
               ->add_dim() = data_shape.dim(i);
        }
      })
      .SetDoc(R"DOC(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q >= 1, gather
slices of `data` into an output tensor of rank q - 1 + r - indices[-1].
Example 1:
  data    = [[0,1],[2,3]]
  indices = [[0,0],[1,1]]
  output  = [0,3]
Example 2:
  data    = [[0,1],[2,3]]
  indices = [[1],[0]]
  output  = [[2,3],[0,1]]
Example 3:
  data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
  indices = [[0,1],[1,0]]
  output  = [[2,3],[4,5]]
Example 4:
  data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
  indices = [[[0,1]],[[1,0]]]
  output  = [[[2,3]],[[4,5]]]
)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(WordConvEmbedding)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr(
          "embedding_size",
          "Integer representing the embedding vector size for each word."
          "If not provide, use the fileter size of conv weight",
          AttributeProto::INT,
          OPTIONAL)
      .Attr(
          "conv_window_size",
          "This operator applies convolution to word from left to right with window equal to conv_window_size and stride to 1."
          "Take word 'example' for example, with conv_window_size equal to 2, conv is applied to [ex],[xa], [am], [mp]..."
          "If not provide, use the first dimension of conv kernal shape.",
          AttributeProto::INT,
          OPTIONAL)
      .Attr(
          "char_embedding_size",
          "Integer representing the embedding vector size for each char."
          "If not provide, use the char embedding size of embedding vector.",
          AttributeProto::INT,
          OPTIONAL)
      .Input(0, "Sequence", "Specify batchs of sequence words to embedding", "T")
      .Input(1, "W", "Specify weights of conv", "T1")
      .Input(2, "B", "Specify bias of conv", "T1")
      .Input(3, "C", "Specify embedding vector of char", "T1")
      .Output(0, "Y", "output", "T1")
      .TypeConstraint(
          "T",
          {"tensor(int32)"},
          "Constrain to tensor(int32).")
      .TypeConstraint(
          "T1",
          {"tensor(float)"},
          "Constrain to tensor(float).")
      .SetDoc(R"DOC(The WordConvEmbedding takes in a batch of sequence words and embed each word to a vector.)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(ROIAlign)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr(
          "spatial_scale",
          "Multiplicative spatial scale factor to translate ROI coordinates "
          "from their input spatial scale to the scale used when pooling, "
          "i.e., spatial scale of the input feature map X relative to the "
          "input image. E.g.; default is 1.0f. ",
          AttributeProto::FLOAT,
          1.f)
      .Attr(
          "pooled_h",
          "default 1; Pooled output Y's height.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Attr(
          "pooled_w",
          "default 1; Pooled output Y's width.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Attr(
          "sampling_ratio",
          "Number of sampling points in the interpolation grid used to compute "
          "the output value of each pooled output bin. If > 0, then exactly "
          "sampling_ratio x sampling_ratio grid points are used. If == 0, then "
          "an adaptive number of grid points are used (computed as "
          "ceil(roi_width / pooled_w), and likewise for height). Default is 0.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "mode",
          "The pooling method. Two modes are supported: 'avg' and 'max'. "
          "Default is 'avg'.",
          AttributeProto::STRING,
          std::string("avg"))
      .Input(0, "X", "Input data tensor from the previous operator; 4-D feature map of shape (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data.", "T")
      .Input(1, "rois", "RoIs (Regions of Interest2) to pool over; rois is 2-D input of shape (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...]. The RoIs' coordinates are in the coordinate system of the input image.", "T")
      .Output(0, "Y", "RoI pooled output, 4-D tesnor of shape (num_rois, C, pooled_h, pooled_w). The r-th batch element Y[r-1] is a pooled feature map corresponding to the r-th RoI X[r-1].", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .SetDoc(R"DOC(Region of Interest (RoI) align operation described in the
  [Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
  RoIAlign consumes an input tensor X and region of interests (rois)
  to apply pooling across each RoI; it produces a 4-D tensor of shape
  (num_rois, C, pooled_h, pooled_w).

  RoIAlign is proposed to avoid the misalignment by removing
  quantizations while converting from original image into feature
  map and from feature map into RoI feature; in each ROI bin,
  the value of the sampled locations are computed directly
  through bilinear interpolation.)DOC");

#ifdef MICROSOFT_INTERNAL
  // register internal ops
  RegisterInternalSchemas();
#endif
}
}  // namespace contrib
}  // namespace onnxruntime
