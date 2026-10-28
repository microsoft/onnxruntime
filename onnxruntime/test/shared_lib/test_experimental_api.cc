// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_experimental_cxx_api.h"

#include "gtest/gtest.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>
#ifdef _WIN32
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#if (defined(_WIN32) || defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)) && !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
namespace {
OrtFileHandle NativeFileHandle(FILE* file) {
#ifdef _WIN32
  return reinterpret_cast<OrtFileHandle>(_get_osfhandle(_fileno(file)));
#else
  return fileno(file);
#endif
}

#ifdef _WIN32
constexpr OrtFileHandle kInvalidFileHandle = nullptr;
#else
constexpr OrtFileHandle kInvalidFileHandle = -1;
#endif
}  // namespace
#endif

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

// Call through typed pointer succeeds and returns the expected sentinel value.
// Uses the throwing accessor (FnOrThrow), which returns a guaranteed-non-null pointer or throws.
// The throw-on-unavailable path is not directly unit-testable: every .inc entry is registered, so a valid
// typed accessor never observes a nullptr. Availability checking is covered by the nullable-accessor tests above.
TEST_F(ExperimentalCApiTest, CallThroughTypedPointer) {
  auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_FnOrThrow(api_);
  ASSERT_NE(fn, nullptr);
  EXPECT_EQ(fn, Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(api_));

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

#if (defined(_WIN32) || defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)) && !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
TEST_F(ExperimentalCApiTest, FileSourcesValidateArgumentsAndAccess) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
  Ort::SessionOptions options;
  EXPECT_FALSE(Ort::Status(fn(nullptr, nullptr, nullptr, 0)).IsOK());
  EXPECT_FALSE(Ort::Status(fn(options, nullptr, nullptr, 1)).IsOK());
  std::unique_ptr<FILE, decltype(&fclose)> file(tmpfile(), fclose);
  ASSERT_NE(file, nullptr);
  const char* name = "weights.data";
  const auto descriptor = NativeFileHandle(file.get());
#ifdef _WIN32
  // Windows duplicates only read rights, even when the original also permits writing.
  EXPECT_TRUE(Ort::Status(fn(options, &name, &descriptor, 1)).IsOK());
#else
  EXPECT_FALSE(Ort::Status(fn(options, &name, &descriptor, 1)).IsOK());
#endif
  const auto invalid = kInvalidFileHandle;
  EXPECT_FALSE(Ort::Status(fn(options, &name, &invalid, 1)).IsOK());
  for (const char* bad : {"../weights.data", "/weights.data", ".", "a/b"}) {
    EXPECT_FALSE(Ort::Status(fn(options, &bad, &invalid, 1)).IsOK());
  }
#ifdef O_PATH
  const int path_only = open("testdata/Pads.bin", O_PATH | O_CLOEXEC);
  ASSERT_GE(path_only, 0);
  EXPECT_FALSE(Ort::Status(fn(options, &name, &path_only, 1)).IsOK());
  EXPECT_EQ(close(path_only), 0);
#endif
}

TEST_F(ExperimentalCApiTest, FileSourcesCloneAndRejectBufferMixing) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
  Ort::SessionOptions options;
  ASSERT_TRUE(Ort::Status(fn(options, nullptr, nullptr, 0)).IsOK());
  auto clone = options.Clone();
  char data[8]{};
  char* buffer = data;
  const ORTCHAR_T* name = ORT_TSTR("weights.data");
  const size_t size = sizeof(data);
  EXPECT_FALSE(Ort::Status(api_->AddExternalInitializersFromFilesInMemory(
                               clone, &name, &buffer, &size, 1))
                   .IsOK());
}

TEST_F(ExperimentalCApiTest, FileSourcesInferAfterHandleAndOptionsRelease) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
  std::ifstream input("testdata/model_with_external_initializers.onnx", std::ios::binary);
  ASSERT_TRUE(input);
  const std::vector<char> graph{std::istreambuf_iterator<char>{input}, {}};
  std::unique_ptr<FILE, decltype(&fclose)> weights(fopen("testdata/Pads.bin", "rb"), fclose);
  ASSERT_NE(weights, nullptr);
  const char* name = "Pads.bin";
  const auto descriptor = NativeFileHandle(weights.get());
  Ort::SessionOptions options;
  ASSERT_TRUE(Ort::Status(fn(options, &name, &descriptor, 1)).IsOK());
  auto clone = options.Clone();
  weights.reset();
  options = Ort::SessionOptions{nullptr};
  const auto invalid = kInvalidFileHandle;
  EXPECT_FALSE(Ort::Status(fn(clone, &name, &invalid, 1)).IsOK());

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "file-sources"};
  Ort::Session session{env, graph.data(), graph.size(), clone};
  clone = Ort::SessionOptions{nullptr};
  Ort::Session reference{env, ORT_TSTR("testdata/model_with_external_initializers.onnx"),
                         Ort::SessionOptions{}};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t shape[]{1, 2};
  float values[]{1.0f, 2.0f};
  auto tensor = Ort::Value::CreateTensor<float>(memory, values, 2, shape, 2);
  const char* inputs[]{"X"};
  const char* outputs[]{"Y"};
  auto expected = reference.Run(Ort::RunOptions{}, inputs, &tensor, 1, outputs, 1);
  auto actual = session.Run(Ort::RunOptions{}, inputs, &tensor, 1, outputs, 1);
  ASSERT_EQ(actual.size(), 1u);
  const auto expected_info = expected[0].GetTensorTypeAndShapeInfo();
  const auto actual_info = actual[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(actual_info.GetShape(), expected_info.GetShape());
  ASSERT_EQ(actual_info.GetElementCount(), expected_info.GetElementCount());
  const auto* expected_data = expected[0].GetTensorData<float>();
  const auto* actual_data = actual[0].GetTensorData<float>();
  for (size_t i = 0; i < actual_info.GetElementCount(); ++i) {
    EXPECT_FLOAT_EQ(actual_data[i], expected_data[i]);
  }
}

TEST_F(ExperimentalCApiTest, FileSourcesNeverFallBackToModelPaths) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
  std::ifstream input("testdata/model_with_external_initializers.onnx", std::ios::binary);
  ASSERT_TRUE(input);
  const std::vector<char> graph{std::istreambuf_iterator<char>{input}, {}};
  ASSERT_FALSE(graph.empty());
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "file-sources"};
  Ort::SessionOptions options;
  ASSERT_TRUE(Ort::Status(fn(options, nullptr, nullptr, 0)).IsOK());
  try {
    Ort::Session session{env, graph.data(), graph.size(), options};
    FAIL() << "Missing registered sources must not fall back to disk";
  } catch (const Ort::Exception& error) {
    EXPECT_NE(std::string(error.what()).find("No registered external initializer file source"),
              std::string::npos);
    EXPECT_NE(std::string(error.what()).find("Pads.bin"), std::string::npos);
  }
}
#endif

#if defined(ORT_MINIMAL_BUILD)
TEST_F(ExperimentalCApiTest, FileSourcesAreNotAdvertisedInMinimalBuilds) {
  EXPECT_EQ(api_->GetExperimentalFunction(
                "OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32"),
            nullptr);
}
#elif (!defined(_WIN32) && !defined(__linux__) && !defined(__ANDROID__) && !defined(__APPLE__)) || defined(DISABLE_EXTERNAL_INITIALIZERS)
TEST_F(ExperimentalCApiTest, FileSourcesRejectUnsupportedBuilds) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
  Ort::SessionOptions options;
  Ort::Status status{fn(options, nullptr, nullptr, 0)};
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_NOT_IMPLEMENTED);
}
#endif
