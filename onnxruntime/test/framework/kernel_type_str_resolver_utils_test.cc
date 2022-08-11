// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_type_str_resolver_utils.h"

#include "gtest/gtest.h"

#include "core/common/string_utils.h"
#include "core/common/parse_string.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/graph/schema_registry.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
static void SplitOpId(const OpIdentifier& op_id,
                      std::string& domain, std::string& op,
                      ONNX_NAMESPACE::OperatorSetVersion& since_version) {
  auto components = utils::SplitString(op_id, ":");
  ASSERT_EQ(components.size(), 3);
  domain = components[0];
  op = components[1];
  since_version = ParseStringWithClassicLocale<ONNX_NAMESPACE::OperatorSetVersion>(components[2]);
}

static void LoadRequiredOpsFromOpSchemas(KernelTypeStrResolver& kernel_type_str_resolver) {
  const auto required_op_ids = kernel_type_str_resolver_utils::GetRequiredOpIdentifiers();
  const auto schema_registry = SchemaRegistryManager{};
  for (auto op_id : required_op_ids) {
    std::string domain{}, op{};
    ONNX_NAMESPACE::OperatorSetVersion since_version{};
    ASSERT_NO_FATAL_FAILURE(SplitOpId(op_id, domain, op, since_version));
    const auto* op_schema = schema_registry.GetSchema(op, since_version, domain);
    ASSERT_NE(op_schema, nullptr);
    ASSERT_STATUS_OK(kernel_type_str_resolver.RegisterOpSchema(*op_schema));
  }
}

TEST(KernelDefTypeStrResolverUtilsTest, VerifyRequiredOpsResolver) {
  KernelTypeStrResolver required_op_kernel_type_str_resolver;
  LoadRequiredOpsFromOpSchemas(required_op_kernel_type_str_resolver);
  // TODO compare
}
}  // namespace onnxruntime::test