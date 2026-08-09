// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <onnx/defs/schema.h>
#include <onnx/defs/function.h>
#include <onnx/defs/shape_inference.h>
#include <pybind11/pybind11.h>
namespace ONNX_NAMESPACE {
void FooShapeInference(InferenceContext& ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}
static const char FooDoc[] = "Foo copies input tensor to the output tensor.";
ONNX_OPERATOR_SET_SCHEMA_EX(
    Foo,
    comExamples,
    "com.examples",
    1,
    false,
    OpSchema()
        .SetDoc(FooDoc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction(FooShapeInference)
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx,
               const OpSchema& schema,
               FunctionProto& functionProto) -> bool {
              if (ctx.getInputType(0) == nullptr) {
                // we cannot create a correct function body without knowing the input type
                return false;
              }
              std::vector<ONNX_NAMESPACE::FunctionBodyHelper::NodeDef> body{
                  {{"Y"}, "Identity", {"X"}}};
              auto func_nodes = ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(body);
              for (const auto& node : func_nodes) {
                auto new_node = functionProto.add_node();
                new_node->CopyFrom(node);
              }
              schema.BuildFunction(functionProto);
              return true;
            }));

}  // namespace ONNX_NAMESPACE

using namespace ONNX_NAMESPACE;
static bool registerOps() {
  auto& d = OpSchemaRegistry::DomainToVersionRange::Instance();
  d.AddDomainToVersion("com.examples", 1, 1);
  auto schema = GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(comExamples, 1, Foo)>();
  RegisterSchema(schema);
  std::cerr << "Successfully registered custom op" << std::endl;
  return true;
}
static bool ret = registerOps();
PYBIND11_MODULE(orttraining_external_custom_ops, m) {
}
