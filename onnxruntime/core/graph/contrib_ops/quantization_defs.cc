// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/contrib_ops/quantization_defs.h"
#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"

namespace ONNX_NAMESPACE {
void RNNShapeInference(InferenceContext& ctx);

void convPoolShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    bool use_dilation, bool require_kernel_shape,
    int input1Idx,
    int input2Idx);
void matmulShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    int input1Idx,
    int input2Idx);

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace contrib {

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::InferenceContext;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;

void ValidateTypeAndShapeForScaleAndZP(ONNX_NAMESPACE::InferenceContext& ctx, int index, ::google::protobuf::int32 expectedType, bool isScalar, int expectedTensorSize) {
  if (ctx.getNumInputs() > static_cast<size_t>(index)) {
    auto data_type = ctx.getInputType(index);
    if (nullptr == data_type) {
      fail_type_inference("Input data type does not match the expected data type");
    }
    if (data_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
        data_type->tensor_type().elem_type() != expectedType) {
      fail_type_inference(
          "Input data type does not match the expected data type. Current data type is ", data_type->tensor_type().elem_type());
    }
  }

  if (hasInputShape(ctx, index)) {
    ONNX_NAMESPACE::TensorShapeProto shape = ctx.getInputType(index)->tensor_type().shape();
    if (isScalar) {
      if (shape.dim_size() != 0) {
        fail_type_inference("Scale and Zero-point must be a scalar");
      }
    } else {
      if (shape.dim_size() != 1) {
        fail_type_inference("Scale and Zero-point must be of rank 1");
      }

      if (shape.dim((int)0).has_dim_value() && shape.dim((int)0).dim_value() != expectedTensorSize) {
        fail_type_inference(
            "Scale and Zero-point must be of rank 1 and the number of elements should be equal to the number of rows of the corresponding input.");
      }
    }
  }
}

std::function<void(OpSchema&)> QLinearMathDocGenerator(const char* name, const char* additionalDocumentation) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
Performs element-wise binary {name} on 8 bit data types (with Numpy-style broadcasting support).

{additionalDocumentation}
)DOC";
    ONNX_NAMESPACE::ReplaceAll(doc, "{name}", name);
    ONNX_NAMESPACE::ReplaceAll(doc, "{additionalDocumentation}", additionalDocumentation);
    schema.SetDoc(doc);
    schema.Input(0, "A", "First operand.", "T");
    schema.Input(
        1,
        "A_scale",
        "Input A's scale. It's a scalar, which means a per-tensor/layer quantization.",
        "tensor(float)");
    schema.Input(
        2,
        "A_zero_point",
        "Input A zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
        "T",
        OpSchema::Optional);
    schema.Input(3, "B", "Second operand.", "T");
    schema.Input(
        4,
        "B_scale",
        "Input B's scale. It's a scalar, which means a per-tensor/layer quantization.",
        "tensor(float)");
    schema.Input(
        5,
        "B_zero_point",
        "Input B zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
        "T",
        OpSchema::Optional);
    schema.Input(
        6,
        "C_scale",
        "Output scale. It's a scalar, which means a per-tensor/layer quantization.",
        "tensor(float)");
    schema.Input(
        7,
        "C_zero_point",
        "Output zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
        "T",
        OpSchema::Optional);
    schema.Output(0, "C", "Result, has same element type as two inputs", "T");
    schema.TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"}, "Constrain input and output types to 8 bit signed and unsigned tensors.");
    schema.TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      auto a_type = ctx.getInputType(0);
      auto b_type = ctx.getInputType(3);

      if (nullptr == a_type || nullptr == b_type ||
          a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
          b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
        fail_type_inference("inputs are expected to have tensor type.");
      }

      // validate scale and zero points
      ValidateTypeAndShapeForScaleAndZP(ctx, 1, ONNX_NAMESPACE::TensorProto::FLOAT, true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 2, a_type->tensor_type().elem_type(), true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 4, ONNX_NAMESPACE::TensorProto::FLOAT, true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 5, b_type->tensor_type().elem_type(), true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 6, ONNX_NAMESPACE::TensorProto::FLOAT, true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 7, a_type->tensor_type().elem_type(), true);

      if (hasInputShape(ctx, 0) && hasInputShape(ctx, 3))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(3)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

void RegisterQuantizationSchemas() {
  static const char* QuantizeLinear_ver1_doc = R"DOC(
The linear quantization operator. It consumes a full precision data, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per 'axis').)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(QuantizeLinear)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr(
          "axis",
          "The axis along which same quantization parameters are applied. It's optional."
          "If it's not specified, it means per-tensor quantization and input 'x_scale' and 'x_zero_point' must be scalars."
          "If it's specified, it means per 'axis' quantization and input 'x_scale' and 'x_zero_point' must be 1-D tensors.",
          AttributeProto::INT,
          false)
      .Input(
          0,
          "x",
          "N-D full precision Input tensor to be quantized.",
          "T1")
      .Input(
          1,
          "y_scale",
          "Scale for doing quantization to get 'y'. It could be a scalar or a 1-D tensor,"
          "which means a per-tensor or per-axis quantization. If it's a 1-D tensor, "
          "its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'.",
          "T1")
      .Input(
          2,
          "y_zero_point",
          "Zero point for doing quantization to get 'y'. It could be a scalar or a 1-D tensor, which means a per-tensor"
          "or per-axis quantization. If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'.",
          "T2")
      .Output(
          0,
          "y",
          "N-D quantized output tensor. It has same shape as input 'x'.",
          "T2")
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)"},
          "Constrain 'x', 'y_scale' to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain 'y_zero_point' and 'y' to 8-bit integer tensors.")
      .SetDoc(QuantizeLinear_ver1_doc)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 2, 0);

        if (!hasInputShape(ctx, 0))
          return;

        auto& input_shape = getInputShape(ctx, 0);
        updateOutputShape(ctx, 0, input_shape);
      });

  static const char* DequantizeLinear_ver1_doc = R"DOC(
The linear dequantization operator. It consumes a quantized data, a scale, a zero point and computes the full precision data.
The dequantization formula is y = (x - x_zero_point) * x_scale.
Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per 'axis').)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(DequantizeLinear)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("axis",
            "The axis along which same quantization parameters are applied. It's optional."
            "If it's not specified, it means per-tensor quantization and input 'x_scale' and 'x_zero_point' must be scalars."
            "If it's specified, it means per 'axis' quantization and input 'x_scale' and 'x_zero_point' must be 1-D tensors.",
            AttributeProto::INT,
            false)
      .Input(
          0,
          "x",
          "N-D quantized Input tensor to be de-quantized.",
          "T1")
      .Input(
          1,
          "x_scale",
          "Scale for input 'x'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-axis quantization."
          "If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'.",
          "T2")
      .Input(
          2,
          "x_zero_point",
          "Zero point for input 'x'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-axis quantization."
          "If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'.",
          "T1")
      .Output(
          0,
          "y",
          "N-D full precision output tensor. It has same shape as input 'x'.",
          "T2")
      .TypeConstraint(
          "T1",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain 'x' and 'x_zero_point' to 8-bit integer tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(float16)", "tensor(float)"},
          "Constrain 'y', 'x_scale' to float tensors.")
      .SetDoc(DequantizeLinear_ver1_doc)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto y_type = ctx.getOutputType(0);
        // only float is supported
        y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);

        if (!hasInputShape(ctx, 0))
          return;

        auto& input_shape = getInputShape(ctx, 0);
        updateOutputShape(ctx, 0, input_shape);
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

  ONNX_CONTRIB_OPERATOR_SCHEMA(MulInteger)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(Performs element-wise binary quantized multiplication (with Numpy-style broadcasting support).
"This operator supports **multidirectional (i.e., Numpy-style) broadcasting**"
The output of this op is the int32 accumulated result of the mul operation

```
C (int32) = (A - A_zero_point) * (B - B_zero_point)
```

)DOC")
      .Input(0, "A", "First operand.", "T")
      .Input(
          1,
          "A_zero_point",
          "Input A zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
          "T",
          OpSchema::Optional)
      .Input(2, "B", "Second operand.", "T")
      .Input(
          3,
          "B_zero_point",
          "Input B zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
          "T",
          OpSchema::Optional)
      .Output(0, "C", "Constrain output to 32 bit tensor", "T1")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"}, "Constrain input types to 8 bit signed and unsigned tensors.")
      .TypeConstraint("T1", {"tensor(int32)"}, "Constrain output types to 32 bit tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto c_type = ctx.getOutputType(0);
        c_type->mutable_tensor_type()->set_elem_type(
            ONNX_NAMESPACE::TensorProto::INT32);

        auto a_type = ctx.getInputType(0);
        auto b_type = ctx.getInputType(3);
        if (nullptr == a_type || nullptr == b_type ||
            a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
            b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
          fail_type_inference("inputs are expected to have tensor type.");
        }

        ValidateTypeAndShapeForScaleAndZP(ctx, 1, a_type->tensor_type().elem_type(), true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 3, b_type->tensor_type().elem_type(), true);

        if (hasInputShape(ctx, 0) && hasInputShape(ctx, 2)) {
          bidirectionalBroadcastShapeInference(
              ctx.getInputType(0)->tensor_type().shape(),
              ctx.getInputType(2)->tensor_type().shape(),
              *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(DynamicQuantizeMatMul)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "A", "N-dimensional matrix A", "T1")
      .Input(1, "B", "N-dimensional matrix B", "T2")
      .Input(
          2,
          "b_scale",
          "Scale of quantized input 'B'. It could be a scalar or a 1-D tensor, "
          "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
          "of elements should be equal to the number of columns of input 'B'.",
          "T1")
      .Input(
          3,
          "b_zero_point",
          "Zero point tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
          "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
          "of elements should be equal to the number of columns of input 'B'.",
          "T2",
          OpSchema::Optional)
      .Input(4,
             "bias",
             "1D input tensor, whose dimension is same as B's last dimension",
             "T1",
             OpSchema::Optional)
      .Output(0, "Y", "Matrix multiply results from A * B", "T1")
      .TypeConstraint(
          "T1",
          {"tensor(float)"},
          "Constrain input A, b_scale and output Y data type as float tensor.")
      .TypeConstraint(
          "T2",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain input B data type to 8-bit integer tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        ONNX_NAMESPACE::matmulShapeInference(ctx, 0, 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MatMulIntegerToFloat)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "A", "N-dimensional matrix A", "T1")
      .Input(1, "B", "N-dimensional matrix B", "T2")
      .Input(
          2,
          "a_scale",
          "Scale of quantized input 'A'. It could be a scalar or a 1-D tensor, "
          "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
          "of elements should be equal to the number of columns of input 'A'.",
          "T3")
      .Input(
          3,
          "b_scale",
          "Scale of quantized input 'B'. It could be a scalar or a 1-D tensor, "
          "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
          "of elements should be equal to the number of columns of input 'B'.",
          "T3")
      .Input(
          4,
          "a_zero_point",
          "Zero point tensor for input 'A'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
          "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
          "of elements should be equal to the number of columns of input 'A'.",
          "T1",
          OpSchema::Optional)
      .Input(
          5,
          "b_zero_point",
          "Zero point tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
          "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
          "of elements should be equal to the number of columns of input 'B'.",
          "T2",
          OpSchema::Optional)
      .Input(
          6,
          "bias",
          "1D input tensor, whose dimension is same as B's last dimension",
          "T3",
          OpSchema::Optional)
      .Output(0, "Y", "Matrix multiply results from A * B", "T3")
      .TypeConstraint(
          "T1",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain input A data type to 8-bit integer tensor.")
      .TypeConstraint(
          "T2",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain input B data type to 8-bit integer tensor.")
      .TypeConstraint(
          "T3",
          {"tensor(float)"},
          "Constrain input a_scale, b_scale and output Y data type as float tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 2, 0);
        ONNX_NAMESPACE::matmulShapeInference(ctx, 0, 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearAdd)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .FillUsing(QLinearMathDocGenerator("addition",
                                         "C = (A_scale * (A - A_zero_point) + B_scale * (B - B_zero_point))/C_scale + C_zero_point"));

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearMul)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .FillUsing(QLinearMathDocGenerator("multiplication",
                                         "C = ((A - A_zero_point) * (B - B_zero_point)) * (A_scale * B_scale)/C_scale + C_zero_point"));

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearReduceMean)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
Computes the mean of the low-precision input tensor's element along the provided axes.
The resulting tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0,
then the resulting tensor have the reduced dimension pruned. The above behavior is similar to numpy,
with the exception that numpy default keepdims to False instead of True.
Input and Output scales and zero points are used to requantize the output in a new range.
This helps to improve accuracy as after ReduceMean operation the range of the output is expected to decrease.

```
"Output = Dequantize(Input) -> ReduceMean on fp32 data -> Quantize(output)",

```
)DOC")
      .Input(0, "data", "An input tensor.", "T")
      .Input(
          1,
          "data_scale",
          "Input scale. It's a scalar, which means a per-tensor/layer quantization.",
          "tensor(float)")
      .Input(
          2,
          "data_zero_point",
          "Input zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
          "T",
          OpSchema::Optional)
      .Input(
          3,
          "reduced_scale",
          "Output scale. It's a scalar, which means a per-tensor/layer quantization.",
          "tensor(float)")
      .Input(
          4,
          "reduced_zero_point",
          "Output zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
          "T",
          OpSchema::Optional)
      .Output(0, "reduced", "Reduced output tensor.", "T")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"}, "Constrain input types to 8 bit signed and unsigned tensors.")
      .Attr(
          "axes",
          "A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor.",
          AttributeProto::INTS)
      .Attr(
          "keepdims",
          "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
          AttributeProto::INT)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        if (!hasNInputShapes(ctx, 1)) {
          return;
        }

        auto data_type = ctx.getInputType(0);
        if (nullptr == data_type || data_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
          fail_type_inference("inputs are expected to have tensor type.");
        }

        // validate scale and zero points
        ValidateTypeAndShapeForScaleAndZP(ctx, 1, ONNX_NAMESPACE::TensorProto::FLOAT, true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 2, data_type->tensor_type().elem_type(), true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 3, ONNX_NAMESPACE::TensorProto::FLOAT, true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 4, data_type->tensor_type().elem_type(), true);

        int64_t keep_dims = 1;
        auto attr_proto = ctx.getAttribute("keepdims");
        if (attr_proto) {
          keep_dims = attr_proto->i();
        }

        auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        int64_t input_ndim = input_shape.dim_size();
        auto output_shape =
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        std::vector<int64_t> axes;
        auto axes_proto = ctx.getAttribute("axes");
        if (axes_proto)
          axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());

        for (size_t i = 0; i < axes.size(); ++i) {
          if (axes[i] < -input_ndim || axes[i] >= input_ndim) {
            fail_shape_inference(
                "axis must be in [-rank, rank-1]. input rank was ", input_ndim);
          }
          if (axes[i] < 0)
            axes[i] += input_ndim;
        }
        // do we need to handle negative axis?
        for (int i = 0; i < input_ndim; ++i) {
          // axes empty means reduce all dim
          if (!axes.empty() &&
              std::find(axes.begin(), axes.end(), i) == axes.end()) {
            auto dim = output_shape->add_dim();
            dim->CopyFrom(input_shape.dim(i));
          } else {
            if (keep_dims == 1) {
              auto dim = output_shape->add_dim();
              dim->set_dim_value(1);
            }
          }
        }
      });

  const char* QLinearLeakyReluDoc_ver1 = R"DOC(
QLinearLeakyRelu takes quantized input data (Tensor), an argument alpha, and quantize parameter for output,
and produces one output data (Tensor<T>) where the function `f(x) = quantize(alpha * dequantize(x)) for dequantize(x) < 0`,
`f(x) = quantize(dequantize(x)) for dequantize(x) >= 0`, is applied to the data tensor elementwise.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearLeakyRelu)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(QLinearLeakyReluDoc_ver1)
      .Attr("alpha", "Coefficient of leakage.", AttributeProto::FLOAT, 0.01f)
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "X_scale",
             "Input X's scale. It's a scalar, which means a per-tensor/layer quantization.",
             "tensor(float)")
      .Input(2, "X_zero_point",
             "Input X's zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
             "T", OpSchema::Optional)
      .Input(3, "Y_scale",
             "Output Y's scale. It's a scalar, which means a per-tensor/layer quantization.",
             "tensor(float)")
      .Input(4, "Y_zero_point",
             "Output Y's zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
             "T", OpSchema::Optional)
      .Output(0, "Y", "Output tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(uint8)", "tensor(int8)"},
          "Constrain input and output types to 8 bit tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  const char* QLinearSigmoidDoc_ver1 = R"DOC(
QLinearSigmoid takes quantized input data (Tensor), and quantize parameter for output, and produces one output data 
(Tensor<T>) where the function `f(x) = quantize(Sigmoid(dequantize(x)))`, is applied to the data tensor elementwise.
Wwhere the function `Sigmoid(x) = 1 / (1 + exp(-x))` )DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearSigmoid)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(QLinearSigmoidDoc_ver1)
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "X_scale",
             "Input X's scale. It's a scalar, which means a per-tensor/layer quantization.",
             "tensor(float)")
      .Input(2, "X_zero_point",
             "Input X's zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
             "T", OpSchema::Optional)
      .Input(3, "Y_scale",
             "Output Y's scale. It's a scalar, which means a per-tensor/layer quantization.",
             "tensor(float)")
      .Input(4, "Y_zero_point",
             "Output Y's zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
             "T", OpSchema::Optional)
      .Output(0, "Y", "Output tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(uint8)", "tensor(int8)"},
          "Constrain input and output types to 8 bit tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(DynamicQuantizeLSTM)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr(
          "direction",
          "Specify if the RNN is forward, reverse, or bidirectional. "
          "Must be one of forward (default), reverse, or bidirectional.",
          AttributeProto::STRING,
          std::string("forward"))
      .Attr(
          "hidden_size",
          "Number of neurons in the hidden layer",
          AttributeProto::INT,
          OPTIONAL_VALUE)
      .Attr(
          "activation_alpha",
          "Optional scaling values used by some activation functions. The values "
          "are consumed in the order of activation functions, for example (f, g, h) "
          "in LSTM. Default values are the same as of corresponding ONNX operators."
          "For example with LeakyRelu, the default alpha is 0.01.",
          AttributeProto::FLOATS,
          OPTIONAL_VALUE)
      .Attr(
          "activation_beta",
          "Optional scaling values used by some activation functions. The values "
          "are consumed in the order of activation functions, for example (f, g, h) "
          "in LSTM. Default values are the same as of corresponding ONNX operators.",
          AttributeProto::FLOATS,
          OPTIONAL_VALUE)
      .Attr(
          "clip",
          "Cell clip threshold. Clipping bounds the elements of a tensor "
          "in the range of [-threshold, +threshold] and is applied to the input "
          "of activations. No clip if not specified.",
          AttributeProto::FLOAT,
          OPTIONAL_VALUE)
      .Attr(
          "activations",
          "A list of 3 (or 6 if bidirectional) activation functions "
          "for input, output, forget, cell, and hidden. The activation functions must "
          "be one of the activation functions specified above. Optional: See the equations "
          "for default if not specified.",
          AttributeProto::STRINGS,
          OPTIONAL_VALUE)
      .Attr(
          "input_forget",
          "Couple the input and forget gates if 1.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Input(
          0,
          "X",
          "The input sequences packed (and potentially padded) into one 3-D "
          "tensor with the shape of `[seq_length, batch_size, input_size]`.",
          "T")
      .Input(
          1,
          "W",
          "The weight tensor for the gates. Concatenation of `W[iofc]` and "
          "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
          "`[num_directions, input_size, 4*hidden_size]`.",
          "T2")
      .Input(
          2,
          "R",
          "The recurrence weight tensor. Concatenation of `R[iofc]` and "
          "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
          "`[num_directions, hidden_size, 4*hidden_size]`.",
          "T2")
      .Input(
          3,
          "B",
          "The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, "
          "and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. This "
          "tensor has shape `[num_directions, 8*hidden_size]`. Optional: If not "
          "specified - assumed to be 0.",
          "T",
          OpSchema::Optional)
      .Input(
          4,
          "sequence_lens",
          "Optional tensor specifying lengths of the sequences in a batch. "
          "If not specified - assumed all sequences in the batch to have "
          "length `seq_length`. It has shape `[batch_size]`.",
          "T1",
          OpSchema::Optional)
      .Input(
          5,
          "initial_h",
          "Optional initial value of the hidden. If not specified - assumed "
          "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
          "T",
          OpSchema::Optional)
      .Input(
          6,
          "initial_c",
          "Optional initial value of the cell. If not specified - assumed "
          "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
          "T",
          OpSchema::Optional)
      .Input(
          7,
          "P",
          "The weight tensor for peepholes. Concatenation of `P[iof]` and "
          "`PB[iof]` (if bidirectional) along dimension 0. It has shape "
          "`[num_directions, 3*hidde_size]`. Optional: If not specified - "
          "assumed to be 0.",
          "T",
          OpSchema::Optional)
      .Input(
          8,
          "W_scale",
          "W's scale. Its size is [num_directions] for per-tensor/layer quantization, "
          "or [num_directions, 4*hidden_size] for per-channel quantization on the axis input_size.",
          "T")
      .Input(
          9,
          "W_zero_point",
          "W's zero point. Its size is [num_directions] for per-tensor/layer quantization, "
          "or [num_directions, 4*hidden_size] for per-channel quantization on the axis input_size.",
          "T2")
      .Input(
          10,
          "R_scale",
          "R's scale. Its size is [num_directions] for per-tensor/layer quantization, "
          "or [num_directions, 4*hidden_size] for per-channel quantization on the axis input_size.",
          "T")
      .Input(
          11,
          "R_zero_point",
          "R's zero point. Its size is [num_directions] for per-tensor/layer quantization, "
          "or [num_directions, 4*hidden_size] for per-channel quantization on the axis input_size.",
          "T2")
      .Output(
          0,
          "Y",
          "A tensor that concats all the intermediate output values of the hidden. "
          "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. ",
          "T",
          OpSchema::Optional,
          true,
          1,
          OpSchema::Differentiable)
      .Output(
          1,
          "Y_h",
          "The last output value of the hidden. It has shape "
          "`[num_directions, batch_size, hidden_size]`.",
          "T",
          OpSchema::Optional,
          true,
          1,
          OpSchema::Differentiable)
      .Output(
          2,
          "Y_c",
          "The last output value of the cell. It has shape "
          "`[num_directions, batch_size, hidden_size]`.",
          "T",
          OpSchema::Optional,
          true,
          1,
          OpSchema::Differentiable)
      .TypeConstraint(
          "T",
          {"tensor(float)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(int32)"},
          "Constrain seq_lens to integer tensor.")
      .TypeConstraint(
          "T2",
          {"tensor(uint8)", "tensor(int8)"},
          "Constrain weights types to 8 bit tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::RNNShapeInference);

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearConcat)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("axis", "Which axis to concat on", AttributeProto::INT)
      .SetDoc(
          "Concatenate a list of tensors into a single tensor."
          "All input tensors must have the same shape, except "
          "for the dimension size of the axis to concatenate on.")
      .Input(0, "Y_scale", "Y's scale.", "TF")
      .Input(1, "Y_zero_point", "Y's zero point.", "T8")
      .Input(2, "inputs", "List of tensors/scale/zero_point for concatenation", "TV", OpSchema::Variadic, false)
      .Output(0, "Y", "Concatenated tensor", "T8")
      .TypeConstraint(
          "T8",
          {"tensor(uint8)", "tensor(int8)"},
          "Constrain input and output types to 8 bit signed and unsigned tensors.")
      .TypeConstraint(
          "TF",
          {"tensor(float)"},
          "Constrain scale types to any float tensor type.")
      .TypeConstraint(
          "TV",
          {"tensor(uint8)", "tensor(int8)", "tensor(float)"},
          "Sequence of (Tensor, Scale, ZeroPoint) tuples. The type is sequence of (T8, TF, T8).")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        auto numInputs = ctx.getNumInputs();
        if (numInputs < 8 || (numInputs - 2) % 3 != 0 ||
            !hasNInputShapes(ctx, static_cast<int>(numInputs))) {
          return;
        }
        auto rank = ctx.getInputType(2)->tensor_type().shape().dim_size();

        auto axisAttr = ctx.getAttribute("axis");
        if (!axisAttr) {
          fail_shape_inference("Required attribute axis is missing");
        }
        int axis = static_cast<int>(axisAttr->i());
        if (rank <= axis || axis < -rank) {
          fail_shape_inference("axis must be in [-rank, rank)");
        }
        if (axis < 0) {
          axis += rank;
        }

        bool all_lengths_known = true;
        int total_length = 0;

        auto* output_shape =
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

        for (int64_t i = 0; i < rank; ++i) {
          output_shape->add_dim();
        }

        for (size_t i = 2; i < numInputs; i += 3) {
          const auto& shape = ctx.getInputType(i)->tensor_type().shape();
          if (shape.dim_size() != rank) {
            fail_shape_inference("All inputs to Concat must have same rank");
          }
          for (int j = 0; j < rank; j++) {
            if (j == axis) {
              if (shape.dim(j).has_dim_value()) {
                total_length += static_cast<int>(shape.dim(j).dim_value());
              } else {
                all_lengths_known = false;
              }
            } else {
              auto& output_dim = *output_shape->mutable_dim(j);
              const auto& input_dim = shape.dim(j);
              mergeInDimensionInfo(input_dim, output_dim, j);
            }
          }
        }

        if (all_lengths_known) {
          output_shape->mutable_dim(axis)->set_dim_value(total_length);
        }
      });
}

}  // namespace contrib
}  // namespace onnxruntime
