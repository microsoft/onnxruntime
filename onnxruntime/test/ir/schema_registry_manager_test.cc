// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/graph/schema_registry.h>
#include "test/providers/provider_test_utils.h"  //For ASSERT_STATUS_OK
#include "gtest/gtest.h"

ONNX_NAMESPACE::OpSchema CreateTestSchema(const char* name, const char* domain, int sinceVersion) {
  return ONNX_NAMESPACE::OpSchema().SetName(name).SinceVersion(sinceVersion).SetDomain(domain).Output(0, "output_1", "docstr for output", "tensor(int32)");
}

using namespace onnxruntime;
TEST(SchemaRegistryManager, search_onnx_op) {
  SchemaRegistryManager manager;
  ASSERT_NE(manager.GetSchema("Gemm", 10, ""), nullptr);
}

TEST(SchemaRegistryManager, search_memcpy_op) {
  SchemaRegistryManager manager;
  ASSERT_NE(manager.GetSchema("MemcpyToHost", 1, ""), nullptr);
}

TEST(SchemaRegistryManager, search_memcpy_op_wrong_version) {
  SchemaRegistryManager manager;
  ASSERT_EQ(manager.GetSchema("MemcpyToHost", 9999, ""), nullptr);
}

#ifndef DISABLE_ML_OPS
TEST(SchemaRegistryManager, search_onnxml_op) {
  SchemaRegistryManager manager;
  ASSERT_NE(manager.GetSchema("ArrayFeatureExtractor", 1, "ai.onnx.ml"), nullptr);
}

TEST(SchemaRegistryManager, search_onnxml_op_wrong_opset_version) {
  SchemaRegistryManager manager;
  ASSERT_EQ(manager.GetSchema("ArrayFeatureExtractor", 99, "ai.onnx.ml"), nullptr);
}
#endif

TEST(SchemaRegistryManager, search_custom_op_wrong_opset_version) {
  SchemaRegistryManager manager;
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::vector<ONNX_NAMESPACE::OpSchema> schema = {CreateTestSchema("Op1", "Domain1", 1)};
  ASSERT_EQ(manager.GetSchema("Op1", 99, "Domain1"), nullptr);
}

TEST(SchemaRegistryManager, OpsetRegTest) {
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();

  // Register op-set version 1 with baseline version 0
  std::vector<ONNX_NAMESPACE::OpSchema> schema = {CreateTestSchema("Op1", "Domain1", 1), CreateTestSchema("Op2", "Domain1", 1)};
  ASSERT_STATUS_OK(registry->RegisterOpSet(schema, "Domain1", 0, 1));

  // Get the schema
  ASSERT_TRUE(registry->GetSchema("Op1", 1, "Domain1")->Name() == "Op1");
  ASSERT_TRUE(registry->GetSchema("Op2", 1, "Domain1")->Name() == "Op2");

  // Getting schema with wrong name, domain, and version will fail
  ASSERT_TRUE(registry->GetSchema("Op1", 1, "WrongDomain") == nullptr);
  ASSERT_TRUE(registry->GetSchema("WrongOp", 1, "Domain1") == nullptr);
  // Fail because this registry doesn't have information for opset2.
  ASSERT_TRUE(registry->GetSchema("Op1", 2, "Domain1") == nullptr);
  ASSERT_TRUE(registry->GetSchema("Op1", 0, "Domain1") == nullptr);

  // Registering a new op-set in the same domain will fail.  This (currently) requires the caller to
  // use multiple registry instances and a registry manager.
  std::vector<ONNX_NAMESPACE::OpSchema> schemaV2 = {CreateTestSchema("Op1", "Domain1", 2)};
  ASSERT_FALSE(registry->RegisterOpSet(schemaV2, "Domain1", 1, 2).IsOK());

  // Registering a second op-set in a different domain should succeed
  std::vector<ONNX_NAMESPACE::OpSchema> schemaDomain2 = {CreateTestSchema("Op2", "Domain2", 1)};
  ASSERT_STATUS_OK(registry->RegisterOpSet(schemaDomain2, "Domain2", 0, 1));
  ASSERT_TRUE(registry->GetSchema("Op1", 1, "Domain1")->Name() == "Op1");
  ASSERT_TRUE(registry->GetSchema("Op2", 1, "Domain2")->Name() == "Op2");

  // Overriding existing op-set schema will fail
  std::vector<ONNX_NAMESPACE::OpSchema> schemaOverride = {CreateTestSchema("Op1", "Domain1", 1)};
  ASSERT_FALSE(registry->RegisterOpSet(schema, "Domain1", 0, 1).IsOK());

  // Create a second registry, combined with the first through a manager
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registry2 = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  SchemaRegistryManager manager;
  manager.RegisterRegistry(registry);
  manager.RegisterRegistry(registry2);

  // Register the second version of the same op-set on the second registry, overriding one operator
  ASSERT_STATUS_OK(registry2->RegisterOpSet(schemaV2, "Domain1", 1, 2));
  //Now the registry1 has: (op1,domain1,version1), (op2,domain1,version1), (op2,domain2,version1)
  //registry2 has:(op1,domain1,version2)
  ASSERT_TRUE(registry2->GetSchema("Op1", 1, "Domain1") == nullptr);
  ASSERT_TRUE(registry2->GetSchema("Op1", 2, "Domain1") != nullptr);
  //Fail because this registery doesn't have the information of opset3
  ASSERT_TRUE(registry2->GetSchema("Op1", 3, "Domain1") == nullptr);

  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registry3 = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  ASSERT_STATUS_OK(registry3->RegisterOpSet(schemaV2, "Domain1", 1, 3));
  //Now it's ok.
  ASSERT_TRUE(registry3->GetSchema("Op1", 3, "Domain1") != nullptr);

  ASSERT_TRUE(manager.GetSchema("Op1", 1, "Domain1")->since_version() == 1);
  ASSERT_TRUE(manager.GetSchema("Op1", 2, "Domain1")->since_version() == 2);
  ASSERT_TRUE(manager.GetSchema("Op2", 1, "Domain1")->since_version() == 1);
  ASSERT_TRUE(manager.GetSchema("Op2", 2, "Domain1")->since_version() == 1);

  // Add a new operator set which is verion 5, with a baseline of version 4, meaning that
  // there is a gap at version 3.
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registryV5 = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  manager.RegisterRegistry(registryV5);
  std::vector<ONNX_NAMESPACE::OpSchema> schemaV5 = {
      CreateTestSchema("Op3", "Domain1", 4),
      CreateTestSchema("Op4", "Domain1", 5),
      CreateTestSchema("Op5", "Domain1", 1)};
  ASSERT_STATUS_OK(registryV5->RegisterOpSet(schemaV5, "Domain1", 4, 5));

  // Query the new version 5 op.  This will be missing for earlier versions
  ASSERT_TRUE(manager.GetSchema("Op4", 5, "Domain1")->since_version() == 5);
  ASSERT_TRUE(manager.GetSchema("Op4", 4, "Domain1") == nullptr);

  // The only schema with SinceVersion < 3 which can be  queried as version 5 are those which are registered on
  // the v5 registry itself.  Those schema may be queried for any version between their sinceVersion and the
  // opset's version.
  ASSERT_TRUE(manager.GetSchema("Op1", 5, "Domain1") == nullptr);
  ASSERT_TRUE(manager.GetSchema("Op3", 5, "Domain1")->since_version() == 4);
  ASSERT_TRUE(manager.GetSchema("Op3", 4, "Domain1")->since_version() == 4);

  // Note that "Op5" has SinceVersion equal to 1, but a V1 operator set was already registered
  // without this operator.  This would normally be invalid, and the registry with the missing
  // operator could trigger the operator lookup to fail.  Version 1 is a special case to allow
  // for experimental operators, and is accomplished by not reducing the targetted version to
  // zero in OnnxRuntimeOpSchemaRegistry::GetSchemaAndHistory.
  // TODO - Consider making the registration algorithm robust to this invalid usage in general
  ASSERT_TRUE(manager.GetSchema("Op5", 5, "Domain1")->since_version() == 1);
  ASSERT_TRUE(manager.GetSchema("Op5", 1, "Domain1")->since_version() == 1);
}