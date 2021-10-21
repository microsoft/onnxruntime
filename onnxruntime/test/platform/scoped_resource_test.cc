// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/scoped_resource.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
namespace {
using TestResourceHandle = int;
constexpr TestResourceHandle k_invalid_handle_value = -1;

// Note: This is not thread-safe! The assumption is that tests in this file are run sequentially.
int g_resource_counter = 0;
int g_next_resource_value = 0;

TestResourceHandle AcquireTestResource() {
  ++g_resource_counter;
  return g_next_resource_value++;
}

void ReleaseTestResource(TestResourceHandle) {
  --g_resource_counter;
}

bool CheckResourceConsistency() {
  return g_resource_counter == 0;
}

struct TestResourceTraits {
  using Handle = TestResourceHandle;
  static Handle GetInvalidHandleValue() noexcept { return k_invalid_handle_value; }
  static void CleanUp(Handle h) noexcept { ReleaseTestResource(h); }
};

using ScopedTestResource = ScopedResource<TestResourceTraits>;

class ScopedResourceTest : public ::testing::Test {
 private:
  void SetUp() override {
    g_resource_counter = 0;
    g_next_resource_value = 0;
  }

  void TearDown() override {
    ASSERT_TRUE(CheckResourceConsistency());
  }
};
}  // namespace

TEST_F(ScopedResourceTest, Basic) {
  ScopedTestResource t{AcquireTestResource()};
  ASSERT_TRUE(t.IsValid());
}

TEST_F(ScopedResourceTest, MoveAssign) {
  ScopedTestResource t1{AcquireTestResource()}, t2{AcquireTestResource()};
  const TestResourceHandle t1_handle = t1.Get();
  t2 = std::move(t1);
  ASSERT_TRUE(!t1.IsValid());
  ASSERT_TRUE(t2.Get() == t1_handle);
}

TEST_F(ScopedResourceTest, MoveConstruct) {
  ScopedTestResource t1{AcquireTestResource()};
  const TestResourceHandle t1_handle = t1.Get();
  ScopedTestResource t2{std::move(t1)};
  ASSERT_TRUE(!t1.IsValid());
  ASSERT_TRUE(t2.Get() == t1_handle);
}

TEST_F(ScopedResourceTest, Reset) {
  ScopedTestResource t{};
  t.Reset(AcquireTestResource());
  ASSERT_TRUE(t.IsValid());
  t.Reset();
  ASSERT_TRUE(!t.IsValid());
}

TEST_F(ScopedResourceTest, Release) {
  TestResourceHandle handle;
  {
    ScopedTestResource t{AcquireTestResource()};
    handle = t.Release();
  }
  ASSERT_NE(handle, k_invalid_handle_value);
  ASSERT_FALSE(CheckResourceConsistency());
  ReleaseTestResource(handle);
}
}  // namespace test
}  // namespace onnxruntime
