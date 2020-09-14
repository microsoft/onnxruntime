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
      .Input(0, "X", "input tensor", "T")
      .Output(0, "Y", "output tensor", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "");
}

}  // namespace audio
}  // namespace onnxruntime
