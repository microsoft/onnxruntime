// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/fallback_cpu_capability.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
namespace {

using fallback_cpu_capability_internal::IsUnsupportedCpuFallbackType;
using fallback_cpu_capability_internal::kUnsupportedCpuFallbackTypes;

consteval bool UnsupportedCpuFallbackTypesAreConstexpr() {
  for (const auto type : kUnsupportedCpuFallbackTypes) {
    if (!IsUnsupportedCpuFallbackType(type)) {
      return false;
    }
  }

  return !IsUnsupportedCpuFallbackType("") &&
         !IsUnsupportedCpuFallbackType("float") &&
         !IsUnsupportedCpuFallbackType("tensor(float16)");
}

static_assert(UnsupportedCpuFallbackTypesAreConstexpr());

TEST(FallbackCpuCapabilityTest, ClassifiesRuntimeTypeNames) {
  const std::string_view unsupported_type = "float8e5m2fnuz";
  const std::string_view supported_type = "tensor(float16)";

  EXPECT_TRUE(IsUnsupportedCpuFallbackType(unsupported_type));
  EXPECT_FALSE(IsUnsupportedCpuFallbackType(supported_type));
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime