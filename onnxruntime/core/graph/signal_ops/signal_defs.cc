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
      .Input(0,
             "input",
             "For real input, the following shape is expected: [batch_idx][n_fft]."
             "For complex input, the following shape is expected: [batch_idx][n_fft][2]." 
             "The final dimension represents the real and imaginary parts of the value."
             "For real multi-dimensional input, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][1]."
             "For complex multi-dimensional input, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]."
             "The first dimension is the batch dimension.",
             "T")
      .Output(0,
              "output",
              "The Fourier Transform of the input vector."
              "If signal_dimN = 1, and onesided is 0, [batch_idx][n_fft][2]"
              "If signal_dimN = 1, and onesided is 1, [batch_idx][floor(n_fft/2)+1][2]" 
              "If signal_dimN = 2, and onesided is 0 and axis = 0, [batch_idx][signal_dim1][signal_dim2][2]"
              "If signal_dimN = 2, and onesided is 0 and axis = 1, [batch_idx][signal_dim1][signal_dim2][2]"
              "If signal_dimN = 2, and onesided is 1 and axis = 0, [batch_idx][floor(signal_dim1/2)+1][signal_dim2][2]"
              "If signal_dimN = 2, and onesided is 1 and axis = 1, [batch_idx][signal_dim1][floor(signal_dim2/2)+1][2]",
              "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          const int64_t batch_ndim = 1;

          auto& input_shape = getInputShape(ctx, 0);
          auto dim_size = static_cast<int64_t>(input_shape.dim_size());
          auto has_component_dimension = dim_size > 2; 

          ONNX_NAMESPACE::TensorShapeProto result_shape_proto = input_shape;
          
          bool is_onesided = static_cast<bool>(getAttribute(ctx, "onesided", 0));
          if (is_onesided) {
              // Since signal_ndim = 1, and multidimensional DFT is not supported,
              // only the single signal dim (1) needs to be updated
              auto n_fft = input_shape.dim(1).dim_value();
              result_shape_proto.mutable_dim(1)->set_dim_value((n_fft >> 1) + 1);
          }
  
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
             "For real input, the following shape is expected: [batch_idx][n_fft]."
             "For complex input, the following shape is expected: [batch_idx][n_fft][2]." 
             "The final dimension represents the real and imaginary parts of the value."
             "For real multi-dimensional input, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][1]."
             "For complex multi-dimensional input, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]."
             "The first dimension is the batch dimension.",
             "T")
      .Output(0,
              "output",
              "The inverse discrete Fourier transform of the input. "
              "If signal_dimN = 1, [batch_idx][n_fft][2]"
              "If signal_dimN = 2 and axis = 0, [batch_idx][signal_dim1][signal_dim2][2]"
              "If signal_dimN = 2 and axis = 1, [batch_idx][signal_dim1][signal_dim2][2]"
              "For all types of input, the last dimension of the output represents the components of a complex number.",
              "T",
              OpSchema::Single,
              true,
              1,
              OpSchema::NonDifferentiable)
      .TypeConstraint(
                "T",
                {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                "Constrain input and output types to float tensors.")
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
          "the real-to-complex Fourier transform satisfies the conjugate symmetry, i.e., X[m, w] = X[m,w]=X[m,n_fft-w]*. "
          "Note if the input or window tensors are complex, then onesided output is not possible. "
          "Enabling onesided with real inputs performs a Real-valued fast Fourier transform (RFFT)."
          "When invoked with real or complex valued input, the default value is 1. "
          "Values can be 0 or 1.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Input(0,
             "signal",
             "Input tensor representing a real or complex valued signal. "
             "For real input, the following shape is expected: [batch_size][signal_length]. "
             "For complex input, the following shape is expected: [batch_size][signal_length][2], where "
             "[batch_size][signal_length][0] represents the real component and [batch_size][signal_length][1] represents the imaginary component of the signal.",
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
              "The inverse fourier transform of the input vector,"
              "using the same format as the input.",
              "T1")
      .TypeConstraint(
          "T1",
          {"tensor(float)",
              "tensor(float16)",
              "tensor(double)",
              "tensor(bfloat16)"},
          "Constrain signal and output to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(int64)"},
          "Constrain scalar length types to int64_t.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
            constexpr int64_t batch_ndim = 1;
            constexpr int64_t component_ndim = 1;

            // Get inputs
            auto& input_shape = getInputShape(ctx, 0);
            auto frame_step = get_scalar_value_from_tensor<int64_t>(ctx.getInputData(1));
            const ONNX_NAMESPACE::TensorShapeProto* window_input = nullptr;
            try {
                window_input = getOptionalInputShape(ctx, 2);
            } catch (...) {
                window_input = nullptr;
            }

            const ONNX_NAMESPACE::TensorShapeProto* frame_length_input = nullptr;
            try {
                frame_length_input = getOptionalInputShape(ctx, 3);
            } catch (...) {
                frame_length_input = nullptr;
            }

            // Determine the size of the DFT based on the 2 optional inputs window and frame_length. One must be set.
            int64_t dft_size = 0;
            if (window_input == nullptr && frame_length_input == nullptr) {
                fail_type_inference("STFT expects to have at least one of these inputs set: [window, frame_length].");
            } else if (window_input != nullptr && window_input->dim_size() > 0 && frame_length_input != nullptr) {
                if (window_input->dim_size() != 1) {
                fail_type_inference("STFT's window input, must have rank = 1.");
                }
                auto window_length = window_input->dim(0).dim_value();
                auto frame_length = get_scalar_value_from_tensor<int64_t>(ctx.getInputData(3));
                if (window_length != frame_length) {
                fail_type_inference("If STFT has both a window input and frame_length specified, the dimension of the window must match the frame_length specified!");
                }
                dft_size = window_length;
            } else if (window_input != nullptr && window_input->dim_size() > 0) {
                if (window_input->dim_size() != 1) {
                fail_type_inference("STFT's window input, must have rank = 1.");
                }
                dft_size = window_input->dim(0).dim_value();
            } else if (frame_length_input != nullptr) {
                dft_size = get_scalar_value_from_tensor<int64_t>(ctx.getInputData(3));
            }

            bool is_onesided = static_cast<bool>(getAttribute(ctx, "onesided", 0));
            if (is_onesided) {
                dft_size = is_onesided ? ((dft_size >> 1) + 1) : dft_size;
            }

            auto signal_size = input_shape.dim(1).dim_value();
            auto n_dfts = static_cast<int64_t>(std::floor((signal_size - dft_size) / static_cast<float>(frame_step)) + 1);

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
      .SetDoc(R"DOC(HannWindow)DOC")
      .Attr("output_datatype",
            "The data type of the output tensor. "
            "Strictly must be one of the types from DataType enum in TensorProto.",
            AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
            static_cast<int64_t>(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT))
      .Input(0,
             "size",
             "A scalar value indicating the length of the Hann Window.",
             "T1")
      .Output(0,
              "output",
              "A Hann Window with length: size.",
              "T2")
      .TypeConstraint("T1", {"tensor(int64)"}, "")
      .TypeConstraint("T2", {"tensor(float)", "tensor(float16)", "tensor(double)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto size = get_scalar_value_from_tensor<int64_t>(ctx.getInputData(0));
        if (size > 0) {
          ONNX_NAMESPACE::TensorShapeProto result_shape;
          result_shape.add_dim()->set_dim_value(size);
          updateOutputShape(ctx, 0, result_shape);
        }

        propagateElemTypeFromAttributeToOutput(ctx, "output_datatype", 0);
      });

  MS_SIGNAL_OPERATOR_SCHEMA(HammingWindow)
      .SetDomain(kMSExperimentalDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(HammingWindow)DOC")
      .Attr("output_datatype",
            "The data type of the output tensor. "
            "Strictly must be one of the types from DataType enum in TensorProto.",
            AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
            static_cast<int64_t>(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT))
      .Input(0,
             "size",
             "A scalar value indicating the length of the Hamming Window.",
             "T1")
      .Output(0,
              "output",
              "A Hamming Window with length: size.",
              "T2")
      .TypeConstraint("T1", {"tensor(int64)"}, "")
      .TypeConstraint("T2", {"tensor(float)", "tensor(float16)", "tensor(double)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto size = get_scalar_value_from_tensor<int64_t>(ctx.getInputData(0));
        if (size > 0) {
          ONNX_NAMESPACE::TensorShapeProto result_shape;
          result_shape.add_dim()->set_dim_value(size);
          updateOutputShape(ctx, 0, result_shape);
        }
        propagateElemTypeFromAttributeToOutput(ctx, "output_datatype", 0);
      });

  MS_SIGNAL_OPERATOR_SCHEMA(BlackmanWindow)
      .SetDomain(kMSExperimentalDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(BlackmanWindow)DOC")
      .Attr("output_datatype",
            "The data type of the output tensor. "
            "Strictly must be one of the types from DataType enum in TensorProto.",
            AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
            static_cast<int64_t>(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT))
      .Input(0,
             "size",
             "A scalar value indicating the length of the Blackman Window.",
             "T1")
      .Output(0,
              "output",
              "A Blackman Window with length: size.",
              "T2")
      .TypeConstraint("T1", {"tensor(int64)"}, "")
      .TypeConstraint("T2", {"tensor(float)", "tensor(float16)", "tensor(double)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto size = get_scalar_value_from_tensor<int64_t>(ctx.getInputData(0));
        if (size > 0) {
          ONNX_NAMESPACE::TensorShapeProto result_shape;
          result_shape.add_dim()->set_dim_value(size);
          updateOutputShape(ctx, 0, result_shape);
        }
        propagateElemTypeFromAttributeToOutput(ctx, "output_datatype", 0);
      });

  MS_SIGNAL_OPERATOR_SCHEMA(MelWeightMatrix)
      .SetDomain(kMSExperimentalDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(MelWeightMatrix)DOC")
      .Attr("output_datatype",
            "The data type of the output tensor. "
            "Strictly must be one of the types from DataType enum in TensorProto.",
            AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
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
      .TypeConstraint("T1", {"tensor(int64)"}, "")
      .TypeConstraint("T2", {"tensor(float)", "tensor(float16)", "tensor(double)"}, "")
      .TypeConstraint("T3", {"tensor(float)", "tensor(float16)", "tensor(double)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto num_mel_bins = get_scalar_value_from_tensor<int64_t>(ctx.getInputData(0));
        auto dft_length = get_scalar_value_from_tensor<int64_t>(ctx.getInputData(1));
        if (num_mel_bins > 0 && dft_length > 0) {
          ONNX_NAMESPACE::TensorShapeProto result_shape;
          // Figure out how to specify one-sided???
          result_shape.add_dim()->set_dim_value(static_cast<int64_t>(std::floor(dft_length / 2.f + 1)));
          result_shape.add_dim()->set_dim_value(num_mel_bins);
          updateOutputShape(ctx, 0, result_shape);
        }
        propagateElemTypeFromAttributeToOutput(ctx, "output_datatype", 0);
      });
}

}  // namespace audio
}  // namespace onnxruntime

#endif