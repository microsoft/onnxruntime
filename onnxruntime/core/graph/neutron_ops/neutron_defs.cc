// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#include "core/graph/constants.h"
#include "core/graph/neutron_ops/neutron_defs.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace onnxruntime {
namespace neutron {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;
using ONNX_NAMESPACE::TensorShapeProto;

void RegisterNeutronSchemas() {
  NEUTRON_OPERATOR_SCHEMA(NeutronGraph)
      .SetDomain(kNeutronDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(Neutron Graph)DOC")
      .Input(0, "Microcode", "", "T2")
      .Input(1, "Weights", "", "T2")
      .Input(2, "Kernels", "", "T2")
      .Input(3, "I", "", "T1", OpSchema::Variadic)
      .Output(0, "Scratch", "", "T2")
      .Output(1, "Profile", "", "T2")
      .Output(2, "Debug", "", "T2")
      .Output(3, "O", "", "T1", OpSchema::Variadic)
      .TypeConstraint("T1", {"tensor(int8)"}, "")
      .TypeConstraint("T2", {"tensor(uint8)"}, "")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext&) {});
}

}  // namespace neutron
}  // namespace onnxruntime
