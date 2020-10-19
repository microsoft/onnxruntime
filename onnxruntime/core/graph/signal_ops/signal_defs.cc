// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(DFT)DOC")
      .Attr("signal_ndim",
            "The number of dimension of the input signal."
            "Values can be 1, 2 or 3.",
            AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
            static_cast<int64_t>(1))
      .Input(0,
          "input",
          "For complex input, the tensor should have one of the following shapes:" 
          "1) signal_ndim = 1 : [batch_idx][Dim1][2]" 
          "2) signal_ndim = 2 : [batch_idx][Dim1][Dim2][2]" 
          "3) signal_ndim = 3 : [batch_idx][Dim1][Dim2][Dim3][2]" 
          "representing the real and imaginary components of complex numbers, " 
          "and the tensor should have signal_ndim + 1 dimensions." 
          "For real input, the tensor should have one of the following shapes:" 
          "1) signal_ndim = 1 : [batch_idx][Dim1]" 
          "2) signal_ndim = 2 : [batch_idx][Dim1][Dim2]" 
          "3) signal_ndim = 3 : [batch_idx][Dim1][Dim2][Dim3]" 
          "and the tensor should have signal_ndim + 1 dimensions." 
          "The first dimension is the batch dimension.",
          "T")
      .Output(0,
              "output",
              "The Fourier Transform of the input vectors.,"
              "using the same format as the input."
              "1) signal_ndim = 1 : [batch_idx][Dim1][2]" 
              "2) signal_ndim = 2 : [batch_idx][Dim1][Dim2][2]" 
              "3) signal_ndim = 3 : [batch_idx][Dim1][Dim2][Dim3][2]",
              "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        size_t ndim = 1;
        auto attr_proto = ctx.getAttribute("signal_ndim");
        if (attr_proto && attr_proto->has_i()) {
          ndim = static_cast<size_t>(attr_proto->i());
        }

        if (ctx.getInputType(0)->tensor_type().has_shape()) {
          auto& input_shape = getInputShape(ctx, 0);
          ONNX_NAMESPACE::TensorShapeProto result_shape = input_shape;

          auto dim_size = input_shape.dim_size();
          if (dim_size == ndim + 1) {                  // real input
            result_shape.add_dim()->set_dim_value(2);  // output is same shape, but with extra dim for 2 values (real/imaginary)
          } else if (dim_size == ndim + 2) {           // complex input, do nothing
          } else {
            fail_shape_inference(
                "the input_shape must have 1 + signal_ndim dimensions for real inputs, or 2 + signal_ndim dimensions for complex input.")
          }
          updateOutputShape(ctx, 0, result_shape);
        }
      });
  ;

  MS_SIGNAL_OPERATOR_SCHEMA(IDFT)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(IDFT)DOC")
      .Attr("signal_ndim",
            "The number of dimension of the input signal."
            "Values can be 1, 2 or 3.",
            AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
            static_cast<int64_t>(1))
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
        size_t ndim = 1;
        auto attr_proto = ctx.getAttribute("signal_ndim");
        if (attr_proto && attr_proto->has_i()) {
          ndim = static_cast<size_t>(attr_proto->i());
        }

        auto& input_shape = getInputShape(ctx, 0);
        ONNX_NAMESPACE::TensorShapeProto result_shape = input_shape;

        auto dim_size = input_shape.dim_size();
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
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(STFT)DOC")
      .Attr("signal_ndim",
            "The number of dimension of the input signal."
            "Values can be 1, 2 or 3.",
            AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
            static_cast<int64_t>(1))
      .Input(0,
             "input",
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
             "hop_length",
             "The number of samples to step between successive DFTs.",
             "T2")
      .Input(3,
             "dft_length",
             "The number of samples to consider in the DFTs.",
             "T2")
      .Output(0,
              "output",
              "The inverse fourier transform of the input vector,"
              "using the same format as the input.",
              "T1")
      .TypeConstraint("T1", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .TypeConstraint("T2", {"tensor(int64)"}, "");

  MS_SIGNAL_OPERATOR_SCHEMA(ISTFT)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(ISTFT)DOC")
      .Attr("signal_ndim",
            "The number of dimension of the input signal."
            "Values can be 1, 2 or 3.",
            AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
            static_cast<int64_t>(1))
      .Input(0,
             "input",
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
             "hop_length",
             "The number of samples to step between successive DFTs.",
             "T2")
      .Input(3,
             "dft_length",
             "The number of samples to consider in the DFTs.",
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
      .SetDomain(kMSDomain)
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
      .SetDomain(kMSDomain)
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
      .SetDomain(kMSDomain)
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
      .SetDomain(kMSDomain)
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
             "num_spectrogram_bins",
             "The number of bins in the spectrogram. FFT size.",
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
        auto num_spectrogram_bins = get_scalar_value_from_tensor<int64_t>(ctx.getInputData(1));
        if (num_mel_bins > 0 && num_spectrogram_bins > 0) {
          ONNX_NAMESPACE::TensorShapeProto result_shape;
          result_shape.add_dim()->set_dim_value(num_mel_bins);
          result_shape.add_dim()->set_dim_value(num_spectrogram_bins);
          updateOutputShape(ctx, 0, result_shape);
        }
        propagateElemTypeFromAttributeToOutput(ctx, "output_datatype", 0);
      });
}

}  // namespace audio
}  // namespace onnxruntime
