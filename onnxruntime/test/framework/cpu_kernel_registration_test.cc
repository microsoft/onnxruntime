// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_kernel_registration.h"

#include "gtest/gtest.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

KernelCreateInfo BuildDisabledEntry() {
  return {};
}

KernelCreateInfo BuildTestEntry() {
  return {};
}

using cpu::registration_internal::IsKernelRegistrationTableValid;

constexpr BuildKernelCreateInfoFn kValidTable[] = {
    BuildDisabledEntry,
    BuildTestEntry,
    BuildDisabledEntry,
};
constexpr BuildKernelCreateInfoFn kMissingDisabledEntryTable[] = {
    BuildTestEntry,
};
constexpr BuildKernelCreateInfoFn kNullFactoryTable[] = {
    BuildDisabledEntry,
    nullptr,
};
constexpr BuildKernelCreateInfoFn kNullDisabledEntryTable[] = {
    nullptr,
};

static_assert(IsKernelRegistrationTableValid(kValidTable, BuildDisabledEntry));
static_assert(!IsKernelRegistrationTableValid(kMissingDisabledEntryTable, BuildDisabledEntry));
static_assert(!IsKernelRegistrationTableValid(kNullFactoryTable, BuildDisabledEntry));
static_assert(!IsKernelRegistrationTableValid(kNullDisabledEntryTable, nullptr));

TEST(CpuKernelRegistrationTest, RegistryIsNotEmpty) {
  const auto kernel_registry = DefaultCpuExecutionProvider()->GetKernelRegistry();

  ASSERT_NE(kernel_registry, nullptr);
  EXPECT_FALSE(kernel_registry->IsEmpty());
}

}  // namespace
}  // namespace onnxruntime::test
