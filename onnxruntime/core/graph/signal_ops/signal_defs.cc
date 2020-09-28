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
  MS_SIGNAL_OPERATOR_SCHEMA(Fft)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(FFT)DOC")
      .Attr("signal_ndim",
          "The number of dimension of the input signal."
          "Values can be 1, 2 or 3.",
          AttributeProto::AttributeType::AttributeProto_AttributeType_INT,
          false)
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
}

}  // namespace audio
}  // namespace onnxruntime
