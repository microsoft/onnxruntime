// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef BUILD_MS_EXPERIMENTAL_OPS

#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/graph/constants.h"
#include "core/graph/signal_ops/signal_defs.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"

#include <cmath>

namespace onnxruntime {
namespace signal {

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;

template <typename T>
static T get_scalar_value_from_tensor(const ONNX_NAMESPACE::TensorProto* t) {
  if (t == nullptr) {
    return T{};
  }

  auto data_type = t->data_type();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<float>(t).at(0));
    case ONNX_NAMESPACE::TensorProto::DOUBLE:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<double>(t).at(0));
    case ONNX_NAMESPACE::TensorProto::INT32:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<int32_t>(t).at(0));
    case ONNX_NAMESPACE::TensorProto::INT64:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<int64_t>(t).at(0));
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
}

inline const ONNX_NAMESPACE::TensorShapeProto* getOptionalInputShape(ONNX_NAMESPACE::InferenceContext& ctx, size_t n) {
  const auto* input_type = ctx.getInputType(n);

  if (input_type == nullptr) {
    return nullptr;
  }

  const auto value_case = input_type->value_case();
  if (value_case != ONNX_NAMESPACE::TypeProto::kTensorType && value_case != ONNX_NAMESPACE::TypeProto::kSparseTensorType) {
    fail_type_inference("Attribute expected to have tensor or sparse tensor type");
  }
  if (value_case == ONNX_NAMESPACE::TypeProto::kTensorType) {
    return &input_type->tensor_type().shape();
  } else {
    return &input_type->sparse_tensor_type().shape();
  }
}

std::function<void(OpSchema&)> CosineSumWindowOpDocGenerator(const char* name) {
  return [name](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
Generates a {name} window as described in the paper https://ieeexplore.ieee.org/document/1455106.
)DOC";
        ReplaceAll(doc, "{name}", name););

    schema.SetDoc(doc);
    schema.Attr("output_datatype",
                "The data type of the output tensor. "
                "Strictly must be one of the values from DataType enum in TensorProto whose values correspond to T2. "
                "The default value is 1 = FLOAT. ",
                AttributeProto::INT,
                static_cast<int64_t>(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT));
    schema.Attr("periodic",
                "If 1, returns a window to be used as periodic function. If 0, return a symmetric window. "
                "When 'periodic' is specified, hann computes a window of length size + 1 and returns the first size points. "
                "The default value is 1. ",
                AttributeProto::INT,
                static_cast<int64_t>(1));
    schema.Input(0,
                 "size",
                 "A scalar value indicating the length of the window.",
                 "T1",
                 OpSchema::Single,
                 true,
                 1,
                 OpSchema::NonDifferentiable);
    schema.Output(0,
                  "output",
                  "A Hann window with length: size. "
                  "The output has the shape: [size].",
                  "T2",
                  OpSchema::Single,
                  true,
                  1,
                  OpSchema::NonDifferentiable);
    schema.TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      // Update the output data type to the output_datatype
      auto output_datatype = getAttribute(ctx, "output_datatype",
                                          static_cast<int64_t>(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT));
      updateOutputElemType(ctx, 0, static_cast<int32_t>(output_datatype));

      if (!hasInputShape(ctx, 0)) {
        // If no shape is available for the input, skip shape inference.
        return;
      }

      const auto* size = ctx.getInputData(0);
      if (size == nullptr) {
        // Size is not available, so return early
        return;
      }

      if (size->dims_size() != 0) {
        fail_shape_inference("size input must be a scalar.");
      }

      auto size_value = get_scalar_value_from_tensor<int64_t>(size);
      if (size_value <= 0) {
        fail_shape_inference("size input must be greater than 0.");
      }

      ONNX_NAMESPACE::TensorShapeProto result_shape;
      result_shape.add_dim()->set_dim_value(size_value);
      updateOutputShape(ctx, 0, result_shape);
    });
  };
}

void RegisterSignalSchemas() {
  MS_SIGNAL_OPERATOR_SCHEMA(DFT)
      .SetDomain(kMSExperimentalDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(DFT)DOC")
      .Attr("onesided",
            "If True (default), only values for half of the fft size are returned because the real-to-complex Fourier transform satisfies the conjugate symmetry."
            "The output tensor will return the first floor(n_fft/2) + 1 values from the DFT."
            "Values can be 0 or 1.",
            AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
            static_cast<int64_t>(0))
      .Attr("axis",
            "The axis on which to perform the DFT. By default this value is set to 0, which corresponds to the first dimension after the batch index."
            "This value must be less than signal_dimN, where signal_dimN is the number of dimensions in the signal.",
            AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
            static_cast<int64_t>(0))
      .Attr("inverse",
            "Whether to perform the inverse discrete fourier transform. By default this value is set to 0, which corresponds to false.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0,
             "input",
             "For real input, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][1]. "
             "For complex input, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]. "
             "The first dimension is the batch dimension. "
             "The following N dimentions correspond to the signal's dimensions. "
             "The final dimension represents the real and imaginary parts of the value in that order.",
             "T1",
             OpSchema::Single,
             true,
             1,
             OpSchema::NonDifferentiable)
      .Input(1,
             "dft_length",
             "The length of the signal."
             "If greater than the axis dimension, the signal will be zero-padded up to dft_length. "
             "If less than the axis dimension, only the first dft_length values will be used as the signal. "
             "It's an optional value. ",
             "T2",
             OpSchema::Optional,
             true,
             1,
             OpSchema::NonDifferentiable)
      .Output(0,
              "output",
              "The Fourier Transform of the input vector."
              "If onesided is 0, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]. "
              "If axis=0 and onesided is 1, the following shape is expected: [batch_idx][floor(signal_dim1/2)+1][signal_dim2]...[signal_dimN][2]. "
              "If axis=1 and onesided is 1, the following shape is expected: [batch_idx][signal_dim1][floor(signal_dim2/2)+1]...[signal_dimN][2]. "
              "If axis=N-1 and onesided is 1, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[floor(signal_dimN/2)+1][2]. "
              "The signal_dim at the specified axis is equal to the dft_length.",
              "T1")
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain scalar length types to int64_t.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        bool is_onesided = static_cast<bool>(getAttribute(ctx, "onesided", 0));
        bool inverse = static_cast<bool>(getAttribute(ctx, "inverse", 0));

        if (inverse && is_onesided) {
          fail_shape_inference("is_onesided and inverse attributes cannot be enabled at the same time");
        }

        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (!hasInputShape(ctx, 0)) {
          // If no shape is available for the input, skip shape inference...
          return;
        }

        // In general the output shape will match the input shape exactly
        // So initialize the output shape with the input shape
        auto& input_shape = getInputShape(ctx, 0);
        ONNX_NAMESPACE::TensorShapeProto result_shape_proto = input_shape;

        // Get the axis where the DFT will be performed.
        auto axis = static_cast<int>(getAttribute(ctx, "axis", 1));
        auto rank = input_shape.dim_size();

        if (!(-rank <= axis && axis < rank)) {
          fail_shape_inference(
              "axis attribute value ",
              axis,
              " is invalid for a tensor of rank ",
              rank);
        }

        auto axis_idx = (axis >= 0 ? axis : axis + rank);

        // If dft_length is specified, then we should honor the shape.
        // Set the output dimension to match the dft_length on the axis.
        // If onesided this will be adjusted later on...
        const ONNX_NAMESPACE::TensorProto* dft_length = nullptr;
        if (ctx.getNumInputs() >= 2 && ctx.getInputType(1) != nullptr) {
          dft_length = ctx.getInputData(1);
          if (dft_length == nullptr) {
            // If we cannot read the dft_length, we cannot infer shape
            // return...
            return;
          }
        }

        if (nullptr != dft_length) {
          if (dft_length->dims_size() != 0) {
            fail_shape_inference("dft_length input must be a scalar.");
          }
          auto dft_length_value = get_scalar_value_from_tensor<int64_t>(dft_length);
          result_shape_proto.mutable_dim(axis_idx)->set_dim_value(dft_length_value);
        }
        // When DFT is onesided, the output shape is half the size of the input shape
        // along the specified axis.
        if (is_onesided) {
          auto axis_dimension = result_shape_proto.dim(axis_idx);
          // We need to update the output shape dimension along the specified axis,
          // but sometimes the dimension will be a free dimension or be otherwise unset.
          // Only perform inference when a input dimension value exists.
          if (axis_dimension.has_dim_value()) {
            auto original_signal_size = axis_dimension.dim_value();
            auto half_signal_size = (original_signal_size >> 1) + 1;
            result_shape_proto.mutable_dim(axis_idx)->set_dim_value(half_signal_size);
          } else {
            // Clear the value and param (which would otherwie be inherited from the input).
            result_shape_proto.mutable_dim(axis_idx)->clear_dim_value();
            result_shape_proto.mutable_dim(axis_idx)->clear_dim_param();
          }
        }

        // Coerce the last dimension to 2.
        auto dim_size = static_cast<int64_t>(result_shape_proto.dim_size());
        auto has_component_dimension = dim_size > 2;

        // This if check is retained in the contrib op and not the official spec for back compat
        if (has_component_dimension) {
          result_shape_proto.mutable_dim(static_cast<int>(dim_size - 1))->set_dim_value(2);
        } else {
          result_shape_proto.add_dim()->set_dim_value(2);
        }

        updateOutputShape(ctx, 0, result_shape_proto);
      });

  MS_SIGNAL_OPERATOR_SCHEMA(IDFT)
      .SetDomain(kMSExperimentalDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(IDFT)DOC")
      .Attr("axis",
            "The axis on which to perform the DFT. By default this value is set to 0, which corresponds to the first dimension after the batch index."
            "This value must be less than signal_dimN, where signal_dimN is the number of dimensions in the signal.",
            AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
            static_cast<int64_t>(0))
      .Input(0,
             "input",
             "For real multi-dimensional input, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][1]."
             "For complex multi-dimensional input, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]."
             "The first dimension is the batch dimension."
             "The final dimension represents the real and imaginary parts of the value.",
             "T1")
      .Input(1,
             "dft_length",
             "The length of the signal."
             "If greater than the axis dimension, the signal will be zero-padded up to dft_length. "
             "If less than the axis dimension, only the first dft_length values will be used as the signal. "
             "It's an optional value. ",
             "T2",
             OpSchema::Optional,
             true,
             1,
             OpSchema::NonDifferentiable)
      .Output(0,
              "output",
              "The inverse discrete Fourier transform of the input. "
              "The signal_dim at the specified axis is equal to the dft_length."
              "The expected shape is [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]"
              "For all types of input, the last dimension of the output represents the components of a complex number.",
              "T1",
              OpSchema::Single,
              true,
              1,
              OpSchema::NonDifferentiable)
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(int64)"},
          "Constrain scalar length types to int64_t.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        const int64_t batch_ndim = 1;

        auto& input_shape = getInputShape(ctx, 0);
        ONNX_NAMESPACE::TensorShapeProto result_shape = input_shape;
        auto dim_size = static_cast<int64_t>(input_shape.dim_size());
        auto has_component_dimension = dim_size > 2;

        if (has_component_dimension) {
          result_shape.mutable_dim(static_cast<int>(dim_size - 1))->set_dim_value(2);
        } else {
          result_shape.add_dim()->set_dim_value(2);
        }

        updateOutputShape(ctx, 0, result_shape);
      });

  MS_SIGNAL_OPERATOR_SCHEMA(STFT)
      .SetDomain(kMSExperimentalDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(STFT)DOC")
      .Attr(
          "onesided",
          "If onesided is 1, only values for w in [0, 1, 2, ..., floor(n_fft/2) + 1] are returned because "
          "the real-to-complex Fourier transform satisfies the conjugate symmetry, i.e., X[m, w] = X[m,w] = "
          "X[m,n_fft-w]*. Note if the input or window tensors are complex, then onesided output is not possible. "
          "Enabling onesided with real inputs performs a Real-valued fast Fourier transform (RFFT)."
          "When invoked with real or complex valued input, the default value is 1. "
          "Values can be 0 or 1.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Input(0,
             "signal",
             "Input tensor representing a real or complex valued signal. "
             "For real input, the following shape is expected: [batch_size][signal_length][1]. "
             "For complex input, the following shape is expected: [batch_size][signal_length][2], where "
             "[batch_size][signal_length][0] represents the real component and [batch_size][signal_length][1] "
             "represents the imaginary component of the signal.",
             "T1",
             OpSchema::Single,
             true,
             1,
             OpSchema::NonDifferentiable)
      .Input(1,
             "frame_step",
             "The number of samples to step between successive DFTs.",
             "T2",
             OpSchema::Single,
             true,
             1,
             OpSchema::NonDifferentiable)
      .Input(2,
             "window",
             "A tensor representing the window that will be slid over the signal."
             "The window must have rank 1 with shape: [window_shape]. "
             "It's an optional value. ",
             "T1",
             OpSchema::Optional,
             true,
             1,
             OpSchema::NonDifferentiable)
      .Input(3,
             "frame_length",
             "A scalar representing the size of the DFT. "
             "It's an optional value.",
             "T2",
             OpSchema::Optional,
             true,
             1,
             OpSchema::NonDifferentiable)
      .Output(0,
              "output",
              "The Short-time Fourier Transform of the signals."
              "If onesided is 1, the output has the shape: [batch_size][frames][dft_unique_bins][2], where "
              "dft_unique_bins is frame_length // 2 + 1 (the unique components of the DFT) "
              "If onesided is 0, the output has the shape: [batch_size][frames][frame_length][2], where frame_length "
              "is the length of the DFT.",
              "T1",
              OpSchema::Single,
              true,
              1,
              OpSchema::NonDifferentiable)
      .TypeConstraint(
          "T1",
          {"tensor(float)",
           "tensor(float16)",
           "tensor(double)",
           "tensor(bfloat16)"},
          "Constrain signal and output to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain scalar length types to int64_t.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // Get signal size
        // The signal size is needed to perform inference because the size of the signal
        // is needed to compute the number of DFTs in the output.
        //
        // 1) Check if shape exists, return if not
        // 2) Get the shape
        // 3) Check if signal dim value exists, return if not
        if (!hasInputShape(ctx, 0)) {
          return;
        }

        auto& input_shape = getInputShape(ctx, 0);
        auto signal_dim = input_shape.dim(1);
        if (!signal_dim.has_dim_value()) {
          return;
        }
        auto signal_size = signal_dim.dim_value();

        // The frame step is a required input.
        // Its value is needed to compute the number output nDFTs, so return early is missing.
        const auto* frame_step = ctx.getInputData(1);
        if (nullptr == frame_step) {
          return;
        }
        auto frame_step_value = get_scalar_value_from_tensor<int64_t>(frame_step);

        // Determine the size of the DFT based on the 2 optional inputs window and frame_length.
        // One must be set.
        int64_t dft_size = -1;
        const ONNX_NAMESPACE::TensorProto* frame_length = nullptr;
        if (ctx.getNumInputs() >= 4 && ctx.getInputType(3) != nullptr) {
          frame_length = ctx.getInputData(3);
          if (frame_length == nullptr) {
            // If we cannot read the frame_length, we cannot infer shape
            // return...
            return;
          }
        }

        const ONNX_NAMESPACE::TensorShapeProto* window_shape = nullptr;
        if (ctx.getNumInputs() >= 3) {
          window_shape = getOptionalInputShape(ctx, 2);
        } else {
          window_shape = nullptr;
        }

        if (window_shape == nullptr && frame_length == nullptr) {
          // STFT expects to have at least one of these inputs set: [window, frame_length],
          // but they may not be available at shape inference time
          return;
        } else if (window_shape != nullptr && frame_length != nullptr) {
          if (frame_length->dims_size() != 0) {
            fail_shape_inference("frame_length input must be scalar.");
          }
          auto frame_length_value = get_scalar_value_from_tensor<int64_t>(frame_length);

          // Ensure that the window length and the dft_length match.
          if (window_shape->dim_size() != 1) {
            fail_shape_inference("window input must have rank = 1.");
          }
          if (window_shape->dim(0).has_dim_value()) {
            auto window_length = window_shape->dim(0).dim_value();
            if (window_length != frame_length_value) {
              fail_type_inference(
                  "If STFT has both a window input and frame_length specified, the dimension of the "
                  "window must match the frame_length specified!");
            }
          }

          dft_size = frame_length_value;
        } else if (window_shape != nullptr) {
          // Ensure that the window length and the dft_length match.
          if (window_shape->dim_size() != 1) {
            fail_shape_inference("window input must have rank = 1.");
          }
          if (window_shape->dim(0).has_dim_value()) {
            dft_size = window_shape->dim(0).dim_value();
          } else {
            // Cannot determine the window size, and there is no frame_length,
            // So shape inference cannot proceed.
            return;
          }
        } else if (frame_length != nullptr) {
          if (frame_length->dims_size() != 0) {
            fail_shape_inference("frame_length input must be scalar.");
          }
          dft_size = get_scalar_value_from_tensor<int64_t>(frame_length);
        }

        bool is_onesided = static_cast<bool>(getAttribute(ctx, "onesided", 0));
        if (is_onesided) {
          dft_size = is_onesided ? ((dft_size >> 1) + 1) : dft_size;
        }

        auto n_dfts = static_cast<int64_t>((signal_size - dft_size) / static_cast<float>(frame_step_value)) + 1;

        // The output has the following shape: [batch_size][frames][dft_unique_bins][2]
        ONNX_NAMESPACE::TensorShapeProto result_shape_proto;
        result_shape_proto.add_dim()->set_dim_value(input_shape.dim(0).dim_value());  // batch size
        result_shape_proto.add_dim()->set_dim_value(n_dfts);
        result_shape_proto.add_dim()->set_dim_value(dft_size);
        result_shape_proto.add_dim()->set_dim_value(2);
        updateOutputShape(ctx, 0, result_shape_proto);
      });

  // Window Functions
  MS_SIGNAL_OPERATOR_SCHEMA(HannWindow)
      .SetDomain(kMSExperimentalDomain)
      .SinceVersion(1)
      .FillUsing(CosineSumWindowOpDocGenerator("Hann"))
      .TypeConstraint(
          "T1",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain the input size to int64_t.")
      .TypeConstraint(
          "T2",
          ONNX_NAMESPACE::OpSchema::all_numeric_types_with_bfloat(),
          "Constrain output types to numeric tensors.")
      .FunctionBody(R"ONNX(
        {
          A0 = Constant <value = float {0.5}>()
          A1 = Constant <value = float {0.5}>()
          A2 = Constant <value = float {0.0}>()
          Zero = Constant <value = float {0.0}>()
          One = Constant <value = float {1.0}>()
          Two = Constant <value = float {2.0}>()
          Tau = Constant <value = float {6.2831853}>()
          Size_FP = Cast <to = 1> (size)
          AngularIncrement = Div (Tau, Size_FP)
          Range = Range (Zero, Size_FP, One)
          RangeAngular = Mul (Range, AngularIncrement)
          TwoRangeAngular = Mul (RangeAngular, Two)
          CosTwoRangeAngular = Cos (TwoRangeAngular)
          A2_Component = Mul (A2, CosTwoRangeAngular)
          CosRangeAngular = Cos (RangeAngular)
          A1_Component = Mul (A1, CosRangeAngular)
          Temp0 = Add (A1_Component, A2_Component)
          Temp1 = Sub (A0, Temp0)
          output = Cast <to : int = @output_datatype> (Temp1)
        }
        )ONNX");

  MS_SIGNAL_OPERATOR_SCHEMA(HammingWindow)
      .SetDomain(kMSExperimentalDomain)
      .SinceVersion(1)
      .FillUsing(CosineSumWindowOpDocGenerator("Hamming"))
      .TypeConstraint(
          "T1",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain the input size to int64_t.")
      .TypeConstraint(
          "T2",
          ONNX_NAMESPACE::OpSchema::all_numeric_types_with_bfloat(),
          "Constrain output types to numeric tensors.")
      .FunctionBody(R"ONNX(
        {
          A0 = Constant <value = float {0.54347826087}>()
          A1 = Constant <value = float {0.45652173913}>()
          A2 = Constant <value = float {0.0}>()
          Zero = Constant <value = float {0.0}>()
          One = Constant <value = float {1.0}>()
          Two = Constant <value = float {2.0}>()
          Tau = Constant <value = float {6.2831853}>()
          Size_FP = Cast <to = 1> (size)
          AngularIncrement = Div (Tau, Size_FP)
          Range = Range (Zero, Size_FP, One)
          RangeAngular = Mul (Range, AngularIncrement)
          TwoRangeAngular = Mul (RangeAngular, Two)
          CosTwoRangeAngular = Cos (TwoRangeAngular)
          A2_Component = Mul (A2, CosTwoRangeAngular)
          CosRangeAngular = Cos (RangeAngular)
          A1_Component = Mul (A1, CosRangeAngular)
          Temp0 = Add (A1_Component, A2_Component)
          Temp1 = Sub (A0, Temp0)
          output = Cast <to : int = @output_datatype> (Temp1)
        }
        )ONNX");

  MS_SIGNAL_OPERATOR_SCHEMA(BlackmanWindow)
      .SetDomain(kMSExperimentalDomain)
      .SinceVersion(1)
      .FillUsing(CosineSumWindowOpDocGenerator("Blackman"))
      .TypeConstraint(
          "T1",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain the input size to int64_t.")
      .TypeConstraint(
          "T2",
          ONNX_NAMESPACE::OpSchema::all_numeric_types_with_bfloat(),
          "Constrain output types to numeric tensors.")
      .FunctionBody(R"ONNX(
        {
          A0 = Constant <value = float {0.42}>()
          A1 = Constant <value = float {0.5}>()
          A2 = Constant <value = float {0.08}>()
          Zero = Constant <value = float {0.0}>()
          One = Constant <value = float {1.0}>()
          Two = Constant <value = float {2.0}>()
          Tau = Constant <value = float {6.2831853}>()
          Size_FP = Cast <to = 1> (size)
          AngularIncrement = Div (Tau, Size_FP)
          Range = Range (Zero, Size_FP, One)
          RangeAngular = Mul (Range, AngularIncrement)
          TwoRangeAngular = Mul (RangeAngular, Two)
          CosTwoRangeAngular = Cos (TwoRangeAngular)
          A2_Component = Mul (A2, CosTwoRangeAngular)
          CosRangeAngular = Cos (RangeAngular)
          A1_Component = Mul (A1, CosRangeAngular)
          Temp0 = Add (A1_Component, A2_Component)
          Temp1 = Sub (A0, Temp0)
          output = Cast <to : int = @output_datatype> (Temp1)
        }
        )ONNX");

  static const char* MelWeightMatrix_ver17_doc = R"DOC(
Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a linearly sampled frequency spectra
(from DFT or STFT) into num_mel_bins frequency information based on the [lower_edge_hertz, upper_edge_hertz] range
on the mel scale.
This function defines the mel scale in terms of a frequency in hertz according to the following formula:

    mel(f) = 2595 * log10(1 + f/700)

In the returned matrix, all the triangles (filterbanks) have a peak value of 1.0.

The returned MelWeightMatrix can be used to right-multiply a spectrogram S of shape [frames, num_spectrogram_bins] of
linear scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram" M of shape [frames, num_mel_bins].
)DOC";

  MS_SIGNAL_OPERATOR_SCHEMA(MelWeightMatrix)
      .SetDomain(kMSExperimentalDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(MelWeightMatrix)DOC")
      .Attr("output_datatype",
            "The data type of the output tensor. "
            "Strictly must be one of the types from DataType enum in TensorProto.",
            ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
            static_cast<int64_t>(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT))
      .Input(0,
             "num_mel_bins",
             "The number of bands in the mel spectrum.",
             "T1")
      .Input(1,
             "dft_length",
             "The size of the FFT.",
             "T1")
      .Input(2,
             "sample_rate",
             "",
             "T1")
      .Input(3,
             "lower_edge_hertz",
             "",
             "T2")
      .Input(4,
             "upper_edge_hertz",
             "",
             "T2")
      .Output(0,
              "output",
              "The MEL Matrix",
              "T3")
      .TypeConstraint(
          "T1",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain to integer tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(float)",
           "tensor(float16)",
           "tensor(double)",
           "tensor(bfloat16)"},
          "Constrain to float tensors")
      .TypeConstraint(
          "T3",
          ONNX_NAMESPACE::OpSchema::all_numeric_types_with_bfloat(),
          "Constrain to any numerical types.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto output_datatype = getAttribute(
            ctx, "output_datatype", static_cast<int64_t>(onnx::TensorProto::DataType::TensorProto_DataType_FLOAT));
        updateOutputElemType(ctx, 0, static_cast<int32_t>(output_datatype));

        if (!hasInputShape(ctx, 0) || !hasInputShape(ctx, 1)) {
          return;
        }

        const auto* num_mel_bins = ctx.getInputData(0);
        const auto* dft_length = ctx.getInputData(1);
        if (nullptr == num_mel_bins || nullptr == dft_length) {
          return;
        }

        int64_t num_mel_bins_value = -1;
        int64_t dft_length_value = -1;
        if (num_mel_bins->dims_size() != 0) {
          fail_shape_inference("num_mel_bins input must be scalar.");
        }
        num_mel_bins_value = get_scalar_value_from_tensor<int64_t>(num_mel_bins);

        if (dft_length->dims_size() != 0) {
          fail_shape_inference("dft_length input must be scalar.");
        }
        dft_length_value = get_scalar_value_from_tensor<int64_t>(dft_length);

        if (num_mel_bins_value > 0 && dft_length_value > 0) {
          ONNX_NAMESPACE::TensorShapeProto result_shape;
          result_shape.add_dim()->set_dim_value(static_cast<int64_t>((dft_length_value >> 1) + 1));
          result_shape.add_dim()->set_dim_value(num_mel_bins_value);
          updateOutputShape(ctx, 0, result_shape);
        }
      });
}

}  // namespace signal
}  // namespace onnxruntime

#endif
