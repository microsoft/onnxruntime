#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/gradient_schema_defs.h"
#include "core/graph/op.h"

namespace onnxruntime {
namespace GradientOps {

std::unordered_map<std::string, GradientOpSchema> GradOpSchemaRegistryHelper::GradientOpRegistry = {
    {"ConvGrad", GradientOpSchema({GO(0), I(0), I(1)}, {GI(0), GI(1), GI(2)})},
    {"MulGrad", GradientOpSchema({GO(0), I(0), I(1)}, {GI(0), GI(1)})},
    {"UnsqueezeGrad", GradientOpSchema({GO(0)}, {GI(0)})},
    {"FlattenGrad", GradientOpSchema({GO(0)}, {GI(0)})},
    {"SinGrad", GradientOpSchema({GO(0), I(0)}, {GI(0)})},
    {"ReluGrad", GradientOpSchema({GO(0), I(0)}, {GI(0)})},
    {"AddGrad", GradientOpSchema({GO(0)}, {GI(0), GI(1)})},
    {"SubGrad", GradientOpSchema({GO(0)}, {GI(0), GI(1)})},
    {"PowGrad", GradientOpSchema({GO(0), I(0), I(1)}, {GI(0), GI(1)})},
    {"MatMulGrad", GradientOpSchema({GO(0), I(0), I(1)}, {GI(0), GI(1)})},
    {"ReduceMeanGrad", GradientOpSchema({GO(0)}, {GI(0)})},
    {"SigmoidGrad", GradientOpSchema({GO(0), I(0)}, {GI(0)})},
    {"SoftmaxGrad", GradientOpSchema({GO(0), I(0)}, {GI(0)})},
};

using namespace ONNX_NAMESPACE;
std::function<void(OpSchema&)> GenGradientSchema(const OpSchema* opSchema) {
  return [=](OpSchema& gradSchema) {
    // Get opschema
    auto iter = GradOpSchemaRegistryHelper::GradientOpRegistry.find(gradSchema.Name());
    if (iter == GradOpSchemaRegistryHelper::GradientOpRegistry.end()) {
      throw NotImplementedException("No implementation yet for this gradient operator");
    }

    auto gradOpSchema = iter->second;
    auto opInputs = opSchema->inputs();
    auto opOutputs = opSchema->outputs();

    int inputCount = 0;
    for (DefsMapping gradOpMapping : gradOpSchema.InputMappings()) {
      int index = gradOpMapping.second;
      auto parameter = gradOpMapping.first == "I" ? opInputs[index] : opOutputs[index];
      auto name = gradOpMapping.first == "GO" ? parameter.GetName() + "Grad" : parameter.GetName();
      auto desc = gradOpMapping.first == "GO" ? "Gradient of" + parameter.GetDescription() : parameter.GetDescription();

      gradSchema.Input(
          inputCount++,
          name,
          desc,
          parameter.GetTypeStr(),
          OpSchema::FormalParameterOption::Optional,
          parameter.GetIsHomogeneous());
    }

    int outputCount = 0;
    for (DefsMapping gradOpMapping : gradOpSchema.OutputMappings()) {
      int index = gradOpMapping.second;
      auto parameter = opInputs[index];

      gradSchema.Output(
          outputCount++,
          "Grad" + parameter.GetName(),
          "Gradient of input:" + parameter.GetDescription(),
          parameter.GetTypeStr(),
          OpSchema::FormalParameterOption::Optional,
          parameter.GetIsHomogeneous());
    }

    // copy over all the attributes schema
    auto attributes = opSchema->attributes();
    for (auto pair : attributes) {
      auto attribute = pair.second;
      gradSchema.Attr(attribute.name, attribute.description, attribute.type, attribute.required);
    }
  };
}

void RegisterGradientSchemas() {
  using namespace ONNX_NAMESPACE;
  auto schema_registry = OpSchemaRegistry::Instance();
  ONNX_CONTRIB_OPERATOR_SCHEMA(SinGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("Sin", 8)));

  ONNX_CONTRIB_OPERATOR_SCHEMA(MulGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("Mul", 8)));

  ONNX_CONTRIB_OPERATOR_SCHEMA(FlattenGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("Flatten", 8)));

  ONNX_CONTRIB_OPERATOR_SCHEMA(UnsqueezeGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("Unsqueeze", 8)));

  ONNX_CONTRIB_OPERATOR_SCHEMA(ConvGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("Conv", 8)));

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReluGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("Relu", 8)));

  ONNX_CONTRIB_OPERATOR_SCHEMA(AddGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("Add", 8)));

  ONNX_CONTRIB_OPERATOR_SCHEMA(MatMulGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("MatMul", 8)));

  ONNX_CONTRIB_OPERATOR_SCHEMA(SubGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("Sub", 8)));

  ONNX_CONTRIB_OPERATOR_SCHEMA(PowGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("Pow", 8)));

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReduceMeanGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("ReduceMean", 8)));

  ONNX_CONTRIB_OPERATOR_SCHEMA(SigmoidGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("Sigmoid", 9)));

  ONNX_CONTRIB_OPERATOR_SCHEMA(SoftmaxGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(8)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.")
      .FillUsing(GenGradientSchema(schema_registry->GetSchema("Softmax", 9)));
}
}  // namespace GradientOps
}  // namespace onnxruntime
