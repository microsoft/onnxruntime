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
using ::ONNX_NAMESPACE::OPTIONAL;
using ::ONNX_NAMESPACE::OpSchema;

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
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
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
a quantized filter, its scale and zero point, and output’s scale and zero point, 
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
      .Input(1, "x_scale", "Scale tensor for input ‘x’. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it’s a 1-D tensor, its number of elements should be equal to the number of channels of input ‘x’.", "T3")
      .Input(2, "x_zero_point", "Zero point tensor for input ‘x’. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it’s a 1-D tensor, its number of elements should be equal to the number of channels of input ‘x’.", "T1")
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
      .Input(4, "w_scale", "Scale tensor for input ‘w’. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it’s a 1-D tensor, its number of elements should be equal to the number of channels of input ‘w’.", "T3")
      .Input(5, "w_zero_point", "Scale tensor for input ‘w’. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it’s a 1-D tensor, its number of elements should be equal to the number of channels of input ‘w’.", "T1")
      .Input(6, "y_scale", "Scale tensor for output ‘y’. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it’s a 1-D tensor, its number of elements should be equal to the number of channels of input ‘y’.", "T3")
      .Input(7, "y_zero_point", "Scale tensor for output ‘y’. It could be a scalar or a 1-D tensor, which means a per-tensor or per-channel quantization. If it’s a 1-D tensor, its number of elements should be equal to the number of channels of input ‘y’.", "T1")
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
          "T1")
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
      .Input(0, "B", "N-dimensional matrix B", "T2")
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
}

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, SampleOp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, ExpandDims);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, AttnLSTM);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, uint8_t, DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, int8_t, DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, QuantizeLinear);

void RegisterContribKernels(std::function<void(KernelCreateInfo&&)> fn) {
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, SampleOp)>());

  // add more kernels here

  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, ExpandDims)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, AttnLSTM)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, IsNaN)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, uint8_t, DequantizeLinear)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, int8_t, DequantizeLinear)>());
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, QuantizeLinear)>());
}
}  // namespace contrib
}  // namespace onnxruntime
