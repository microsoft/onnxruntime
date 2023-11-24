// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_type_str_resolver_utils.h"

#include <iostream>
#include <sstream>

#include "gtest/gtest.h"

#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/graph/schema_registry.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {

static Status LoadLayoutTransformationRequiredOpsFromOpSchemas(KernelTypeStrResolver& kernel_type_str_resolver) {
  const auto required_op_ids = kernel_type_str_resolver_utils::GetLayoutTransformationRequiredOpIdentifiers();
  const auto schema_registry = SchemaRegistryManager{};
  for (const auto& op_id : required_op_ids) {
    const auto* op_schema = schema_registry.GetSchema(std::string{op_id.op_type}, op_id.since_version,
                                                      std::string{op_id.domain});
    ORT_RETURN_IF(op_schema == nullptr, "Failed to get op schema.");
    ORT_RETURN_IF_ERROR(kernel_type_str_resolver.RegisterOpSchema(*op_schema));
  }
  return Status::OK();
}

TEST(KernelTypeStrResolverUtilsTest, VerifyLayoutTransformationRequiredOpsResolver) {
  KernelTypeStrResolver expected_resolver;
  ASSERT_STATUS_OK(LoadLayoutTransformationRequiredOpsFromOpSchemas(expected_resolver));

  KernelTypeStrResolver actual_resolver;
  ASSERT_STATUS_OK(
      kernel_type_str_resolver_utils::AddLayoutTransformationRequiredOpsToKernelTypeStrResolver(actual_resolver));

#if !defined(DISABLE_CONTRIB_OPS)
  ASSERT_EQ(actual_resolver.GetOpKernelTypeStrMap(), expected_resolver.GetOpKernelTypeStrMap());
#else   // !defined(DISABLE_CONTRIB_OPS)
  // check that each element of expected_resolver is present and equivalent in actual_resolver
  const auto& expected_op_kernel_type_str_map = expected_resolver.GetOpKernelTypeStrMap();
  const auto& actual_op_kernel_type_str_map = actual_resolver.GetOpKernelTypeStrMap();

  for (const auto& [expected_op_id, expected_kernel_type_str_map] : expected_op_kernel_type_str_map) {
    const auto actual_op_kernel_type_str_map_it = actual_op_kernel_type_str_map.find(expected_op_id);
    ASSERT_NE(actual_op_kernel_type_str_map_it, actual_op_kernel_type_str_map.end());
    ASSERT_EQ(actual_op_kernel_type_str_map_it->second, expected_kernel_type_str_map);
  }
#endif  // !defined(DISABLE_CONTRIB_OPS)
}

// run this test manually to output a hard-coded byte array.
// update AddLayoutTransformationRequiredOpsToKernelTypeStrResolver in
// onnxruntime/core/framework/kernel_type_str_resolver_utils.cc
TEST(KernelTypeStrResolverUtilsTest, DISABLED_PrintExpectedLayoutTransformationRequiredOpsResolverByteArray) {
#if defined(DISABLE_CONTRIB_OPS)
  FAIL() << "Contrib ops must be enabled.";
#else   // defined(DISABLE_CONTRIB_OPS)
  KernelTypeStrResolver expected_resolver;
  ASSERT_STATUS_OK(LoadLayoutTransformationRequiredOpsFromOpSchemas(expected_resolver));

  flatbuffers::DetachedBuffer buffer;
  gsl::span<const uint8_t> buffer_span;
  ASSERT_STATUS_OK(kernel_type_str_resolver_utils::SaveKernelTypeStrResolverToBuffer(expected_resolver,
                                                                                     buffer, buffer_span));

  constexpr size_t kBytesPerLine = 16;
  std::ostringstream os;
  os << std::hex << std::setfill('0')
     << "  constexpr uint8_t kLayoutTransformationRequiredOpsKernelTypeStrResolverBytes[] = {\n      ";
  for (size_t i = 0; i < buffer_span.size(); ++i) {
    os << "0x" << std::setw(2) << static_cast<int32_t>(buffer_span[i]) << ",";
    if (i < buffer_span.size() - 1) {
      os << ((i % kBytesPerLine == kBytesPerLine - 1) ? "\n      " : " ");
    }
  }
  os << "\n  };\n";

  std::cout << os.str();
#endif  // defined(DISABLE_CONTRIB_OPS)
}

}  // namespace onnxruntime::test
