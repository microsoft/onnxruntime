#include <iostream>
#include <onnx/defs/schema.h>
#include <onnx/defs/function.h>
#include <onnx/defs/shape_inference.h>
#include <pybind11/pybind11.h>

static const char FooDoc[] = "Foo copies input tensor to the output tensor.";

using namespace ONNX_NAMESPACE;
bool registerOps() {
  std::cout << "In registerOps" << std::endl;
  auto &d = OpSchemaRegistry::DomainToVersionRange::Instance();
  std::cout << "AddDomainToVersion" << std::endl;
  d.AddDomainToVersion("com.examples", 1, 1);
  OpSchema schema;
  schema.SetName("Foo")
    .SetDomain("com.examples")
    .SinceVersion(1)
    .SetDoc(FooDoc)
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float)", "tensor(int32)", "tensor(float16)"},
        "Constrain input and output types to signed numeric tensors.")
    .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
    })
    .SetContextDependentFunctionBodyBuilder(
        [](const FunctionBodyBuildContext& ctx,
            const OpSchema& schema,
            FunctionProto& functionProto) -> bool {
          if (ctx.getInputType(0) == nullptr) {
            // we cannot create a correct function body without knowing the input type
            return false;
          }
          std::vector<ONNX_NAMESPACE::FunctionBodyHelper::NodeDef> body{
                                            {{"Y"}, "Identity", { "X"}}
                                            };
          auto func_nodes = ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(body);
          for (const auto& node : func_nodes) {
            auto new_node = functionProto.add_node();
            new_node->CopyFrom(node);
          }
          schema.BuildFunction(functionProto);
          return true;
        });
  std::cout << "RegisterSchema" << std::endl;
  RegisterSchema(schema);
  std::cout << "Successfully registered custom op" << std::endl;
  return true;
}

PYBIND11_MODULE(orttraining_external_custom_ops, m) {
  m.def("register_custom_ops", &registerOps, "Register custom operators.");
}
