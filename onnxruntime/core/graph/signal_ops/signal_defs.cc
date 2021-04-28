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
      .Input(0,
          "input",
          "For complex input, the following shape is expected: [batch_idx][n_fft][2]" 
          "The final dimension represents the real and imaginary parts of the value."
          "For real input, the following shape is expected: [batch_idx][n_fft]"
          "The first dimension is the batch dimension.",
          "T")
      .Output(0,
              "output",
              "The Fourier Transform of the input vector."
              "If onesided is 1, [batch_idx][floor(n_fft/2)+1][2]" 
              "If onesided is 0, [batch_idx][n_fft][2]",
              "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        int64_t ndim = 1;

        bool is_onesided = true;
        auto attr_proto = ctx.getAttribute("onesided");
        if (attr_proto && attr_proto->has_i()) {
          is_onesided = static_cast<bool>(attr_proto->i());
        }

        if (ctx.getInputType(0)->tensor_type().has_shape()) {
          auto& input_shape = getInputShape(ctx, 0);
          ONNX_NAMESPACE::TensorShapeProto result_shape = input_shape;

          if (is_onesided) {
            auto n_fft = input_shape.dim(1).dim_value();
            result_shape.mutable_dim(1)->set_dim_value((n_fft >> 1) + 1);
          }

          auto dim_size = static_cast<int64_t>(input_shape.dim_size());
          if (dim_size == ndim + 1) {                  // real input
            result_shape.add_dim()->set_dim_value(2);  // output is same shape, but with extra dim for 2 values (real/imaginary)
          } else if (dim_size == ndim + 2) {           // complex input, do nothing
          } else {
            fail_shape_inference(
                "the input_shape must [batch_idx][n_fft] for real values or [batch_idx][n_fft][2] for complex values.")
          }
          updateOutputShape(ctx, 0, result_shape);
        }
      });
  ;

  MS_SIGNAL_OPERATOR_SCHEMA(IDFT)
      .SetDomain(kMSExperimentalDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(IDFT)DOC")
      .Input(0,
             "input",
             "A complex signal of dimension signal_ndim."
             "The last dimension of the tensor should be 2,"
             "representing the real and imaginary components of complex numbers,"
             "and should have at least signal_ndim + 2 dimensions."
             "The first dimension is the batch dimension.",
             "T")
      .Output(0,
              "output",
              "The inverse fourier transform of the input vector,"
              "using the same format as the input.",
              "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        int64_t ndim = 1;
        auto attr_proto = ctx.getAttribute("signal_ndim");
        if (attr_proto && attr_proto->has_i()) {
          ndim = static_cast<size_t>(attr_proto->i());
        }

        auto& input_shape = getInputShape(ctx, 0);
        ONNX_NAMESPACE::TensorShapeProto result_shape = input_shape;

        auto dim_size = static_cast<int64_t>(input_shape.dim_size());
        if (dim_size == ndim + 1) {                  // real input
          result_shape.add_dim()->set_dim_value(2);  // output is same shape, but with extra dim for 2 values (real/imaginary)
        } else if (dim_size == ndim + 2) {           // complex input, do nothing
        } else {
          fail_shape_inference(
              "the input_shape must have 1 + signal_ndim dimensions for real inputs, or 2 + signal_ndim dimensions for complex input.")
        }

        updateOutputShape(ctx, 0, result_shape);
      });

  MS_SIGNAL_OPERATOR_SCHEMA(STFT)
      .SetDomain(kMSExperimentalDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(STFT)DOC")
      .Attr("onesided",
            "If True (default), only values for half of the fft size are returned because the real-to-complex Fourier transform satisfies the conjugate symmetry."
            "The output tensor will return the first floor(n_fft/2) + 1 values from the DFT."
            "Values can be 0 or 1.",
            AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
            static_cast<int64_t>(1))
      .Input(0,
             "signal",
             "A complex signal of dimension signal_ndim."
             "The last dimension of the tensor should be 2,"
             "representing the real and imaginary components of complex numbers,"
             "and should have at least signal_ndim + 2 dimensions."
             "The first dimension is the batch dimension.",
             "T1")
      .Input(1,
             "window",
             "A tensor representing the window that will be slid over the input signal.",
             "T1",
             OpSchema::FormalParameterOption::Optional)
      .Input(2,
             "frame_length",  // frame_length, fft_length, pad_mode
             "Size of the fft.",
             "T2",
             OpSchema::FormalParameterOption::Optional)
      .Input(3,
             "frame_step",
             "The number of samples to step between successive DFTs.",
             "T2")
      .Output(0,
              "output",
              "The inverse fourier transform of the input vector,"
              "using the same format as the input.",
              "T1")
      .TypeConstraint("T1", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .TypeConstraint("T2", {"tensor(int64)"}, "");

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