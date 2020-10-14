// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/signal_ops/signal_defs.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace onnxruntime {
namespace signal {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;

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
             "A complex signal of dimension signal_ndim."
             "The last dimension of the tensor should be 2,"
             "representing the real and imaginary components of complex numbers,"
             "and should have at least signal_ndim + 2 dimensions."
             "The first dimension is the batch dimension.",
             "T")
      .Output(0,
              "output",
              "The fourier transform of the input vector,"
              "using the same format as the input.",
              "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "");

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
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "");

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
      .TypeConstraint("T1", {"tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "")
      .TypeConstraint("T2", {"tensor(float)", "tensor(float16)", "tensor(double)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "");

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
      .TypeConstraint("T1", {"tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "")
      .TypeConstraint("T2", {"tensor(float)", "tensor(float16)", "tensor(double)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "");
    
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
      .TypeConstraint("T1", {"tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "")
      .TypeConstraint("T2", {"tensor(float)", "tensor(float16)", "tensor(double)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "");

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
      .TypeConstraint("T1", {"tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "")
      .TypeConstraint("T2", {"tensor(float)", "tensor(float16)", "tensor(double)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "")
      .TypeConstraint("T3", {"tensor(float)", "tensor(float16)", "tensor(double)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}, "");
}

}  // namespace audio
}  // namespace onnxruntime
