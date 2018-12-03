// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/contrib_ops.h"

#include "core/graph/constants.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"

#include "./cpu/attnlstm/attn_lstm_schema_defs.h"

namespace onnxruntime {
namespace contrib {
using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::OpSchema;
using ::ONNX_NAMESPACE::OPTIONAL;

void RegisterContribSchemas() {
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

  ONNX_CONTRIB_OPERATOR_SCHEMA(IsNaN)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "input", "T1")
      .Output(0, "Y", "output", "T2")
      .TypeConstraint(
          "T1",
          ONNX_NAMESPACE::OpSchema::numeric_types_for_math_reduction(),
          "Constrain to any numeric tensor type. If the dtype attribute is not provided this must be a valid output type.")
      .TypeConstraint(
          "T2",
          {"tensor(bool)"},
          "Constrain outputs to boolean tensor")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
      .SetDoc(R"DOC(Returns which elements of the input are NaN.)DOC");

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
 rounding halfway cases away from zero. Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per 'axis').)DOC");

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
 Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per 'axis').)DOC");

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
should be equal to the number of columns of input 'b'.)DOC")
      .Input(0, "a", "N-dimensional quantized matrix a", "T1")
      .Input(1, "a_scale", "scale of quantized input a", "tensor(float)")
      .Input(2, "a_zero_point", "zero point of quantized input a", "T1")
      .Input(3, "b", "N-dimensional quantized matrix b", "T2")
      .Input(4, "b_scale", "scale of quantized input b", "tensor(float)")
      .Input(5, "b_zero_point", "zero point of quantized input b", "T2")
      .Input(6, "y_scale", "scale of quantized output y", "tensor(float)")
      .Input(7, "y_zero_point", "zero point of quantized output y", "T3")
      .Output(0, "y", "Quantized matrix multiply results from a * b", "T3")
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input a and its zero point data types as 8-bit integer tensor")
      .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain input b and its zero point data types as 8-bit integer tensor")
      .TypeConstraint("T3", {"tensor(int8)", "tensor(uint8)"}, "Constrain output y and its zero point data types as 8-bit integer tensor.");

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
It means they must be either scalars (per tensor) or 1-D tensors (per channel).)DOC")
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
      .Input(1, "x_scale", "Scale tensor for input 'x'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of channels of input 'x'.", "T3")
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
          "T1")
      .Input(4, "w_scale", "Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of channels of input 'w'.", "T3")
      .Input(5, "w_zero_point", "Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of channels of input 'w'.", "T1")
      .Input(6, "y_scale", "Scale tensor for output 'y'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of channels of input 'y'.", "T3")
      .Input(7, "y_zero_point", "Scale tensor for output 'y'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of channels of input 'y'.", "T1")
      .Input(8, "B", "Optional 1D bias to be added to the convolution, has size of M.", "T2", OpSchema::Optional)
      .Output(
          0,
          "y",
          "Output data tensor that contains the result of the "
          "convolution. The output dimensions are functions "
          "of the kernel size, stride size, and pad lengths.",
          "T1")
      .TypeConstraint(
          "T1",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain input, filter, and output types to 8-bit integer tensors.")
      .TypeConstraint("T2", {"tensor(int32)", "tensor(uint32)"}, "Constrain bias type to 32-bit integer tensor.")
      .TypeConstraint("T3", {"tensor(float)"}, "Constrain scale of input, filter and output to float tensor.")
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
          static_cast<int64_t>(1));

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
      .Input(2, "z", "Padding value (zero_point normally), it's optional and default value is 0.", "T1", OpSchema::Optional)
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
          static_cast<int64_t>(1));

  ONNX_CONTRIB_OPERATOR_SCHEMA(MatMulInteger)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
 The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.)DOC")
      .Input(0, "A", "N-dimensional matrix A", "T1")
      .Input(1, "B", "N-dimensional matrix B", "T2")
      .Output(0, "Y", "Matrix multiply results from A * B", "T3")
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input A data types as 8-bit integer tensor")
      .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain input B data types as 8-bit integer tensor")
      .TypeConstraint("T3",
                      {"tensor(int32)", "tensor(uint32)"},
                      "Constrain output Y data types as 32-bit integer tensor."
                      "T3 must be tensor(uint32) when both T1 and T2 are tensor(uint8),"
                      "or must be tensor(int32) when either T1 or T2 is tensor(int8).");

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
Pruning away boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding boxes are supplied as [y1, x1, y2, x2],
where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box corners and the coordinates can be provided
as normalized (i.e., lying in the interval [0, 1]) or absolute.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system;
thus translating or reflections of the coordinate system result in the same boxes being selected by the algorithm.
The output of this operation is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the gather operation.)DOC")
      .Input(0, "boxes", "An input tensor. 2D tensor with shape [num_boxes, 4]", "T1")
      .Input(1, "scores", "An input tensor. 1D tensor with shape [num_boxes]", "T1")
      .Output(0, "selected_indices", "selected indices from the boxes tensor.", "T2")
      .Output(
          1,
          "valid_outputs",
          "Optional. A 0-D integer tensor representing the number of valid elements in selected_indices, with the valid elements appearing first.",
          "T2",
          OpSchema::Optional)
      .TypeConstraint("T1", {"tensor(float)"}, "Constrain input type to float tensor.")
      .TypeConstraint("T2",
                      {"tensor(int32)"},
                      "Constrain output data type to 32-bit integer tensor.")
      .Attr(
          "max_output_size",
          "Integer representing the maximum number of boxes to be selected by non max suppression.",
          AttributeProto::INT)
      .Attr(
          "iou_threshold",
          "Float representing the threshold for deciding whether boxes overlap too much with respect to IOU. Value range [0, 1]. The default is 0.0",
          AttributeProto::FLOAT,
          static_cast<float>(0.0f))
      .Attr(
          "score_threshold",
          "Float tensor representing the threshold for deciding when to remove boxes based on score.",
          AttributeProto::FLOAT)
      .Attr(
          "pad_to_max_output_size",
          "Optional. 1(true) - the output selected_indices is padded to be of length max_output_size. Defaults to 0(false).",
          AttributeProto::INT,
          OPTIONAL)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto selected_indices_type = ctx.getOutputType(0)->mutable_tensor_type();
        selected_indices_type->set_elem_type(::onnx::TensorProto_DataType::TensorProto_DataType_INT32);

        // If pad_to_max_output_size is set to 1, the output(0) selected_indices will has a fixed shape [max_output_size].
        auto pad_to_max_output_size = ctx.getAttribute("pad_to_max_output_size");
        if (pad_to_max_output_size && 1 == pad_to_max_output_size->i()) {
          auto max_output_size = ctx.getAttribute("max_output_size")->i();
          selected_indices_type
              ->mutable_shape()
              ->add_dim()
              ->set_dim_value(max_output_size);
        }

        // valid_outputs is optional, shape is [1]
        auto num_outputs = ctx.getNumOutputs();
        if (num_outputs > 1) {
          auto valid_outputs_shape = ctx.getOutputType(1)->mutable_tensor_type();
          valid_outputs_shape->set_elem_type(::onnx::TensorProto_DataType::TensorProto_DataType_INT32);
          valid_outputs_shape
              ->mutable_shape()
              ->add_dim()
              ->set_dim_value(1);
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(StringNormalizer)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "Strings to normalize", "T")
      .Output(0, "Y", "Normalized strings", "T")
      .TypeConstraint(
          "T",
          {"tensor(string)"},
          "Input/Output is a string tensor")
      .Attr(
          "casechangeaction",
          "string enum that cases output to be lowercased/uppercases/unchanged. Valid values are \"LOWER\", \"UPPER\", \"NONE\"",
          AttributeProto::STRING)
      .Attr(
          "is_case_sensitive",
          "Boolean. Whether the identification of stop words in X is case-sensitive.",
          AttributeProto::INT)
      .Attr(
          "stopwords",
          "List of stop words",
          AttributeProto::STRINGS,
          OPTIONAL)
      .Attr(
          "locale",
          "Environment dependent string that denotes the locale according to which output strings needs to be upper/lowercased. Default en_US",
          AttributeProto::STRING,
          OPTIONAL)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
        output_elem_type->set_elem_type(ONNX_NAMESPACE::TensorProto::STRING);
      })
      .SetDoc(R"DOC([optional] Step1: Remove elements in X if they matches any of stop words so that output tensor may not contain any stop word. This operator only accepts [C]- and [1, C]-tensor. If all elements in X are dropped, the output will be the default value of string tensor with shape [1] if input shape is [C] and shape [1, 1] if input shape is [1, C]. 
[optional] Step2: Lower all characters (if action is LOWER) in X or capitalize them (when action is UPPER))DOC");
}

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, SampleOp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, ExpandDims);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, AttnLSTM);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, uint8_t, DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, int8_t, DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, QuantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, string, StringNormalizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, NonMaxSuppression);

void RegisterContribKernels(std::function<void(KernelCreateInfo&&)> fn) {
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, SampleOp)>());

  // add more kernels here

  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, ExpandDims)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, AttnLSTM)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, IsNaN)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, uint8_t, DequantizeLinear)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, int8_t, DequantizeLinear)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, QuantizeLinear)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, string, StringNormalizer)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, NonMaxSuppression)>());
}
}  // namespace contrib
}  // namespace onnxruntime
