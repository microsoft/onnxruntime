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

static Status LoadRequiredOpsFromOpSchemas(KernelTypeStrResolver& kernel_type_str_resolver) {
  const auto required_op_ids = kernel_type_str_resolver_utils::GetLayoutTransformationRequiredOpIdentifiers();
  const auto schema_registry = SchemaRegistryManager{};
  for (auto op_id : required_op_ids) {
    const auto* op_schema = schema_registry.GetSchema(std::string{op_id.op_type}, op_id.since_version,
                                                      std::string{op_id.domain});
    ORT_RETURN_IF(op_schema == nullptr, "Failed to get op schema.");
    ORT_RETURN_IF_ERROR(kernel_type_str_resolver.RegisterOpSchema(*op_schema));
  }
  return Status::OK();
}

TEST(KernelTypeStrResolverUtilsTest, VerifyRequiredOpsResolver) {
  KernelTypeStrResolver expected_resolver;
  ASSERT_STATUS_OK(LoadRequiredOpsFromOpSchemas(expected_resolver));

  KernelTypeStrResolver actual_resolver;
  ASSERT_STATUS_OK(
      kernel_type_str_resolver_utils::AddLayoutTransformationRequiredOpsToKernelTypeStrResolver(actual_resolver));

  ASSERT_EQ(actual_resolver.GetOpKernelTypeStrMap(), expected_resolver.GetOpKernelTypeStrMap());
}

// run this test manually to output a hard-coded byte array
TEST(KernelTypeStrResolverUtilsTest, DISABLED_PrintExpectedLayoutTransformationRequiredOpsResolverByteArray) {
  KernelTypeStrResolver expected_resolver;
  ASSERT_STATUS_OK(LoadRequiredOpsFromOpSchemas(expected_resolver));

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
}

}  // namespace onnxruntime::test