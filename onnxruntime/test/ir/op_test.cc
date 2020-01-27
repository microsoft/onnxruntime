// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/op.h"
#include <iostream>
#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"
#include "core/graph/schema_registry.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "test/test_environment.h"

using namespace ONNX_NAMESPACE;

#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

#define OPERATOR_SCHEMA UNUSED ONNX_OPERATOR_SCHEMA

namespace onnxruntime {
namespace test {
TEST(FormalParamTest, Success) {
  OpSchema::FormalParameter p("input", "desc: integer input", "tensor(int32)");
  EXPECT_EQ("input", p.GetName());
  EXPECT_EQ("tensor(int32)", p.GetTypeStr());
  EXPECT_EQ("desc: integer input", p.GetDescription());
  // TODO: change onnx to make formal parameter construction self-contain.
  //EXPECT_EQ(Utils::DataTypeUtils::ToType("tensor(int32)"), *p.GetTypes().begin());
}

TEST(FeatureVectorizerTest, TraditionalMlOpTest) {
  Model model("traditionalMl", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  // Case: A traditional ml graph.
  //                           SouceNode
  //                              |
  //                       node_1(CastMap)
  //                      (tensor(float))
  //                             |
  //                    node_5 (FeatureVectorizer)
  //                              |
  //                           SinkNode

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // Type: tensor(float)
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);

  // Type: map(int64,float);
  TypeProto map_int64_float;
  auto map_type = map_int64_float.mutable_map_type();
  map_type->set_key_type(TensorProto::INT64);
  auto map_value_type = map_type->mutable_value_type()->mutable_tensor_type();
  map_value_type->set_elem_type(TensorProto::FLOAT);
  map_value_type->mutable_shape();

  NodeArg* input_arg1 = new NodeArg("node_1_in_1", &map_int64_float);
  inputs.clear();
  inputs.push_back(input_arg1);
  NodeArg* output_arg1 = new NodeArg("node_1_out_1", &tensor_float);
  outputs.clear();
  outputs.push_back(output_arg1);
  graph.AddNode("node_1", "CastMap", "node 1", inputs, outputs, nullptr, kMLDomain);

  inputs.clear();
  inputs.push_back(output_arg1);

  NodeArg* output_arg4 = new NodeArg("node_4_out_1", &tensor_float);
  outputs.clear();
  outputs.push_back(output_arg4);
  graph.AddNode("node_4", "FeatureVectorizer", "node 4", inputs, outputs, nullptr, kMLDomain);
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  delete input_arg1;
  delete output_arg1;
  delete output_arg4;
}

TEST(OpRegistrationTest, OpRegTest) {
  OPERATOR_SCHEMA(__TestOpReg)
      .SetDoc("Op Registration Basic Test.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Input(1, "input_2", "docstr for input_2.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  const OpSchema* op_schema = OpSchemaRegistry::Schema("__TestOpReg");
  EXPECT_TRUE(nullptr != op_schema);
  EXPECT_EQ(op_schema->inputs().size(), 2);
  EXPECT_EQ(op_schema->inputs()[0].GetName(), "input_1");
  EXPECT_EQ(op_schema->inputs()[0].GetTypes().size(), 1);
  EXPECT_EQ(**op_schema->inputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(int32)")), "tensor(int32)");
  EXPECT_EQ(op_schema->inputs()[1].GetName(), "input_2");
  EXPECT_EQ(op_schema->inputs()[1].GetTypes().size(), 1);
  EXPECT_EQ(**op_schema->inputs()[1].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(int32)")), "tensor(int32)");
  EXPECT_EQ(op_schema->outputs().size(), 1);
  EXPECT_EQ(op_schema->outputs()[0].GetName(), "output_1");
  EXPECT_EQ(op_schema->outputs()[0].GetTypes().size(), 1);
  EXPECT_EQ(**op_schema->outputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(int32)")), "tensor(int32)");
}

ONNX_NAMESPACE::OpSchema CreateTestSchema(const char* name, const char* domain, int sinceVersion) {
  return ONNX_NAMESPACE::OpSchema().SetName(name).SinceVersion(sinceVersion).SetDomain(domain).Output(0, "output_1", "docstr for output", "tensor(int32)");
}

TEST(OpRegistrationTest, OpsetRegTest) {
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();

  // Register op-set version 1 with baseline version 0
  std::vector<ONNX_NAMESPACE::OpSchema> schema = {CreateTestSchema("Op1", "Domain1", 1), CreateTestSchema("Op2", "Domain1", 1)};
  EXPECT_TRUE(registry->RegisterOpSet(schema, "Domain1", 0, 1).IsOK());

  // Get the schema
  EXPECT_TRUE(registry->GetSchema("Op1", 1, "Domain1")->Name() == "Op1");
  EXPECT_TRUE(registry->GetSchema("Op2", 1, "Domain1")->Name() == "Op2");

  // Getting schema with wrong name, domain, and version will fail
  EXPECT_TRUE(registry->GetSchema("Op1", 1, "WrongDomain") == nullptr);
  EXPECT_TRUE(registry->GetSchema("WrongOp", 1, "Domain1") == nullptr);
  EXPECT_TRUE(registry->GetSchema("Op1", 2, "Domain1") == nullptr);
  EXPECT_TRUE(registry->GetSchema("Op1", 0, "Domain1") == nullptr);

  // Registering a new op-set in the same domain will fail.  This (currently) requires the caller to
  // use multiple registry instances and a registry manager.
  std::vector<ONNX_NAMESPACE::OpSchema> schemaV2 = {CreateTestSchema("Op1", "Domain1", 2)};
  EXPECT_FALSE(registry->RegisterOpSet(schemaV2, "Domain1", 1, 2).IsOK());

  // Registering an op-set with schema in a different domain than the op-set will fail
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> temp_reg = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  EXPECT_FALSE(temp_reg->RegisterOpSet(schema, "WrongDomain", 0, 1).IsOK());

  // Registering a second op-set in a different domain should succeed
  std::vector<ONNX_NAMESPACE::OpSchema> schemaDomain2 = {CreateTestSchema("Op2", "Domain2", 1)};
  EXPECT_TRUE(registry->RegisterOpSet(schemaDomain2, "Domain2", 0, 1).IsOK());
  EXPECT_TRUE(registry->GetSchema("Op1", 1, "Domain1")->Name() == "Op1");
  EXPECT_TRUE(registry->GetSchema("Op2", 1, "Domain2")->Name() == "Op2");

  // Overriding existing op-set schema will fail
  std::vector<ONNX_NAMESPACE::OpSchema> schemaOverride = {CreateTestSchema("Op1", "Domain1", 1)};
  EXPECT_FALSE(registry->RegisterOpSet(schema, "Domain1", 0, 1).IsOK());

  // Create a second registry, combined with the first through a manager
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registry2 = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  SchemaRegistryManager manager;
  manager.RegisterRegistry(registry);
  manager.RegisterRegistry(registry2);

  // Register the second version of the same op-set on the second registry, overriding one operator
  EXPECT_TRUE(registry2->RegisterOpSet(schemaV2, "Domain1", 1, 2).IsOK());
  EXPECT_TRUE(manager.GetSchema("Op1", 1, "Domain1")->since_version() == 1);
  EXPECT_TRUE(manager.GetSchema("Op1", 2, "Domain1")->since_version() == 2);
  EXPECT_TRUE(manager.GetSchema("Op2", 1, "Domain1")->since_version() == 1);

  // Op2 is provided only in opset v1, and in the first registry.  The absence of Op2 in the second
  // registry will trigger the first registry to be queried using V1 rather than V2 here.
  EXPECT_TRUE(manager.GetSchema("Op2", 2, "Domain1")->since_version() == 1);

  // Add a new operator set which is verion 5, with a baseline of version 4, meaning that
  // there is a gap at version 3.
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registryV5 = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  manager.RegisterRegistry(registryV5);
  std::vector<ONNX_NAMESPACE::OpSchema> schemaV5 = {
      CreateTestSchema("Op3", "Domain1", 4),
      CreateTestSchema("Op4", "Domain1", 5),
      CreateTestSchema("Op5", "Domain1", 1)};
  EXPECT_TRUE(registryV5->RegisterOpSet(schemaV5, "Domain1", 4, 5).IsOK());

  // Query the new version 5 op.  This will be missing for earlier versions
  EXPECT_TRUE(manager.GetSchema("Op4", 5, "Domain1")->since_version() == 5);
  EXPECT_TRUE(manager.GetSchema("Op4", 4, "Domain1") == nullptr);

  // The only schema with SinceVersion < 3 which can be  queried as version 5 are those which are registered on
  // the v5 registry itself.  Those schema may be queried for any version between their sinceVersion and the
  // opset's version.
  EXPECT_TRUE(manager.GetSchema("Op1", 5, "Domain1") == nullptr);
  EXPECT_TRUE(manager.GetSchema("Op3", 5, "Domain1")->since_version() == 4);
  EXPECT_TRUE(manager.GetSchema("Op3", 4, "Domain1")->since_version() == 4);

  // Note that "Op5" has SinceVersion equal to 1, but a V1 operator set was already registered
  // without this operator.  This would normally be invalid, and the registry with the missing
  // operator could trigger the operator lookup to fail.  Version 1 is a special case to allow
  // for experimental operators, and is accomplished by not reducing the targetted version to
  // zero in OnnxRuntimeOpSchemaRegistry::GetSchemaAndHistory.
  // TODO - Consider making the registration algorithm robust to this invalid usage in general
  EXPECT_TRUE(manager.GetSchema("Op5", 5, "Domain1")->since_version() == 1);
  EXPECT_TRUE(manager.GetSchema("Op5", 1, "Domain1")->since_version() == 1);
}

TEST(OpRegistrationTest, TypeConstraintTest) {
  OPERATOR_SCHEMA(__TestTypeConstraint)
      .SetDoc("Op with Type Constraint.")
      .Input(0, "input_1", "docstr for input_1.", "T")
      .Input(1, "input_2", "docstr for input_2.", "T")
      .Output(0, "output_1", "docstr for output_1.", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to floats.");
  const OpSchema* op_schema = OpSchemaRegistry::Schema("__TestTypeConstraint");
  EXPECT_TRUE(nullptr != op_schema);
  EXPECT_EQ(op_schema->inputs().size(), 2);
  EXPECT_EQ(op_schema->inputs()[0].GetName(), "input_1");
  EXPECT_EQ(op_schema->inputs()[0].GetTypes().size(), 3);
  EXPECT_EQ(**op_schema->inputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(float16)")), "tensor(float16)");
  EXPECT_EQ(**op_schema->inputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(float)")), "tensor(float)");
  EXPECT_EQ(**op_schema->inputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(double)")), "tensor(double)");

  EXPECT_EQ(op_schema->inputs()[1].GetName(), "input_2");
  EXPECT_EQ(op_schema->inputs()[1].GetTypes().size(), 3);
  EXPECT_EQ(**op_schema->inputs()[1].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(float16)")), "tensor(float16)");
  EXPECT_EQ(**op_schema->inputs()[1].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(float)")), "tensor(float)");
  EXPECT_EQ(**op_schema->inputs()[1].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(double)")), "tensor(double)");

  EXPECT_EQ(op_schema->outputs().size(), 1);
  EXPECT_EQ(op_schema->outputs()[0].GetName(), "output_1");
  EXPECT_EQ(op_schema->outputs()[0].GetTypes().size(), 3);
  EXPECT_EQ(**op_schema->outputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(float16)")), "tensor(float16)");
  EXPECT_EQ(**op_schema->outputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(float)")), "tensor(float)");
  EXPECT_EQ(**op_schema->outputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(double)")), "tensor(double)");
}

TEST(OpRegistrationTest, AttributeDefaultValueTest) {
  OPERATOR_SCHEMA(__TestAttrDefaultValue)
      .SetDoc("Op with attributes that have default values")
      .Attr("my_attr_int", "attr with default value of 99.", AttrType::AttributeProto_AttributeType_INT, int64_t(99))
      .Attr("my_attr_float", "attr with default value of 0.99.", AttrType::AttributeProto_AttributeType_FLOAT, float(0.99))
      .Attr("my_attr_string", "attr with default value of \"99\".", AttrType::AttributeProto_AttributeType_STRING, std::string("99"));
  const OpSchema* op_schema = OpSchemaRegistry::Schema("__TestAttrDefaultValue");
  EXPECT_TRUE(nullptr != op_schema);
  EXPECT_EQ(op_schema->attributes().size(), 3);

  auto attr_int = op_schema->attributes().find("my_attr_int")->second;
  EXPECT_EQ(attr_int.name, "my_attr_int");
  EXPECT_EQ(attr_int.type, AttrType::AttributeProto_AttributeType_INT);
  EXPECT_FALSE(attr_int.required);
  EXPECT_EQ(attr_int.default_value.name(), "my_attr_int");
  EXPECT_TRUE(attr_int.default_value.has_i());
  EXPECT_EQ(attr_int.default_value.i(), 99LL);

  auto attr_float = op_schema->attributes().find("my_attr_float")->second;
  EXPECT_EQ(attr_float.name, "my_attr_float");
  EXPECT_EQ(attr_float.type, AttrType::AttributeProto_AttributeType_FLOAT);
  EXPECT_FALSE(attr_float.required);
  EXPECT_EQ(attr_float.default_value.name(), "my_attr_float");
  EXPECT_TRUE(attr_float.default_value.has_f());
  EXPECT_EQ(attr_float.default_value.f(), 0.99f);

  auto attr_string = op_schema->attributes().find("my_attr_string")->second;
  EXPECT_EQ(attr_string.name, "my_attr_string");
  EXPECT_EQ(attr_string.type, AttrType::AttributeProto_AttributeType_STRING);
  EXPECT_FALSE(attr_string.required);
  EXPECT_EQ(attr_string.default_value.name(), "my_attr_string");
  EXPECT_TRUE(attr_string.default_value.has_s());
  EXPECT_EQ(attr_string.default_value.s(), "99");
}

TEST(OpRegistrationTest, AttributeDefaultValueListTest) {
  OPERATOR_SCHEMA(__TestAttrDefaultValueList)
      .SetDoc("Op with attributes that have default list of values.")
      .Attr("my_attr_ints", "attr with default value of [98, 99, 100].", AttrType::AttributeProto_AttributeType_INTS, std::vector<int64_t>{int64_t(98), int64_t(99), int64_t(100)})
      .Attr("my_attr_floats", "attr with default value of [0.98, 0.99, 1.00].", AttrType::AttributeProto_AttributeType_FLOATS, std::vector<float>{float(0.98), float(0.99), float(1.00)})
      .Attr("my_attr_strings", "attr with default value of [\"98\", \"99\", \"100\"].", AttrType::AttributeProto_AttributeType_STRINGS, std::vector<std::string>{"98", "99", "100"});
  const OpSchema* op_schema = OpSchemaRegistry::Schema("__TestAttrDefaultValueList");
  EXPECT_TRUE(nullptr != op_schema);
  EXPECT_EQ(op_schema->attributes().size(), 3);

  auto attr_ints = op_schema->attributes().find("my_attr_ints")->second;
  EXPECT_EQ(attr_ints.name, "my_attr_ints");
  EXPECT_EQ(attr_ints.type, AttrType::AttributeProto_AttributeType_INTS);
  EXPECT_FALSE(attr_ints.required);
  EXPECT_EQ(attr_ints.default_value.name(), "my_attr_ints");
  int size = attr_ints.default_value.ints_size();
  EXPECT_EQ(size, 3);
  std::vector<int64_t> expected_ints = {98LL, 99LL, 100LL};
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(attr_ints.default_value.ints(i), expected_ints[i]);
  }

  auto attr = op_schema->attributes().find("my_attr_floats")->second;
  EXPECT_EQ(attr.name, "my_attr_floats");
  EXPECT_EQ(attr.type, AttrType::AttributeProto_AttributeType_FLOATS);
  EXPECT_FALSE(attr.required);
  EXPECT_EQ(attr.default_value.name(), "my_attr_floats");
  size = attr.default_value.floats_size();
  EXPECT_EQ(size, 3);
  std::vector<float> expected_floats = {0.98f, 0.99f, 1.00f};
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(attr.default_value.floats(i), expected_floats[i]);
  }

  auto attr2 = op_schema->attributes().find("my_attr_strings")->second;
  EXPECT_EQ(attr2.name, "my_attr_strings");
  EXPECT_EQ(attr2.type, AttrType::AttributeProto_AttributeType_STRINGS);
  EXPECT_FALSE(attr2.required);
  EXPECT_EQ(attr2.default_value.name(), "my_attr_strings");
  size = attr2.default_value.strings_size();
  EXPECT_EQ(size, 3);
  std::vector<std::string> expected_strings = {"98", "99", "100"};
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(attr2.default_value.strings(i), expected_strings[i]);
  }
}

}  // namespace test
}  // namespace onnxruntime
