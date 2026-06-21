// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_experimental_c_api.h"

#include "gtest/gtest.h"

class ExperimentalCApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ASSERT_NE(api_, nullptr);
  }

  const OrtApi* api_ = nullptr;
};

// Null name returns nullptr
TEST_F(ExperimentalCApiTest, NullNameReturnsNull) {
  OrtExperimentalFnPtr fn = api_->GetExperimentalFunction(nullptr);
  EXPECT_EQ(fn, nullptr);
}

// Unknown name returns nullptr
TEST_F(ExperimentalCApiTest, UnknownNameReturnsNull) {
  OrtExperimentalFnPtr fn = api_->GetExperimentalFunction("NonExistentFunction");
  EXPECT_EQ(fn, nullptr);
}

// Empty string returns nullptr
TEST_F(ExperimentalCApiTest, EmptyNameReturnsNull) {
  OrtExperimentalFnPtr fn = api_->GetExperimentalFunction("");
  EXPECT_EQ(fn, nullptr);
}

// Known name resolves to non-null (C-style lookup)
TEST_F(ExperimentalCApiTest, KnownNameResolvesC) {
  OrtExperimentalFnPtr fn =
      api_->GetExperimentalFunction(kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName);
  EXPECT_NE(fn, nullptr);
}

// Known name resolves to non-null (C++-style accessor).
// The typed accessor's nullptr path (unknown name) is covered transitively via the C-side
// UnknownNameReturnsNull test, which exercises the same underlying GetExperimentalFunction lookup.
TEST_F(ExperimentalCApiTest, KnownNameResolvesCpp) {
  auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(api_);
  EXPECT_NE(fn, nullptr);
}

// Call through typed pointer succeeds and returns the expected sentinel value
TEST_F(ExperimentalCApiTest, CallThroughTypedPointer) {
  auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(api_);
  ASSERT_NE(fn, nullptr);

  int64_t result = 0;
  auto status = Ort::Status{fn(&result)};
  ASSERT_TRUE(status.IsOK()) << "ExperimentalApiTest returned failure status: " << status.GetErrorMessage();
  EXPECT_EQ(result, 12345);
}

// The same name looked up twice returns the same pointer
TEST_F(ExperimentalCApiTest, ConsistentLookup) {
  OrtExperimentalFnPtr fn1 =
      api_->GetExperimentalFunction(kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName);
  OrtExperimentalFnPtr fn2 =
      api_->GetExperimentalFunction(kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName);
  EXPECT_EQ(fn1, fn2);
}
