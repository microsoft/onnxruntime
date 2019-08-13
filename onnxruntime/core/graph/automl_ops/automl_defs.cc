// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/automl_ops/automl_defs.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace onnxruntime {
namespace automl {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL;

void RegisterAutoMLSchemas() {

  static const char* DateTimeTransformer_ver1_doc = R"DOC(
    DateTimeTransformer accepts a single scalar int64 tensor, constructs
    an instance of std::chrono::system_clock::time_point and passes it as an argument
    to Microsoft::DateTimeFeaturizer which is a part of a shared library.
    It returns an instance of TimePoint class.
  )DOC";

  MS_AUTOML_OPERATOR_SCHEMA(DateTimeTransformer)
      .SinceVersion(1)
      .SetDomain(kMSAutoMLDomain)
      .SetDoc(DateTimeTransformer_ver1_doc)
      .Input(0, "X",
             "The input represents a number of seconds passed since the epoch, suitable to properly construct"
             "an instance of std::chrono::system_clock::time_point",
             "T1")
      .Output(0, "Y", "The output which is a Microsoft::DateTimeFeaturizer::TimePoint structure", "T2")
      .TypeConstraint(
          "T1",
          {"tensor(int64)"},
          "Constrain input type to int64 scalar tensor.")
      .TypeConstraint(
          "T2",
          {"opaque(com.microsoft.automl,DateTimeFeaturizer_TimePoint)"},
          "Constrain output type to an AutoML specific Microsoft::Featurizers::TimePoint type"
          "currently not part of ONNX standard. When it becomes a part of the standard we will adjust this"
          "kernel definition and move it to ONNX repo");
}
}  // namespace automl
}  // namespace onnxruntime
