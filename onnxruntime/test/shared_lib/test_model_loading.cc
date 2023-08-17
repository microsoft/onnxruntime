// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"
#include "core/common/narrow.h"
#include "test/util/include/asserts.h"
#include <fstream>
#include "test_fixture.h"
#include "file_util.h"

#include "gmock/gmock.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

// disable for minimal build with no exceptions as it will always attempt to throw in that scenario
#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_NO_EXCEPTIONS)
TEST(CApiTest, model_from_array) {
  const char* model_path = "testdata/matmul_1.onnx";
  std::vector<char> buffer;
  {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file)
      ORT_THROW("Error reading model");
    buffer.resize(narrow<size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer.data(), buffer.size()))
      ORT_THROW("Error reading model");
  }

#if (!ORT_MINIMAL_BUILD)
  bool should_throw = false;
#else
  bool should_throw = true;
#endif

  auto create_session = [&](Ort::SessionOptions& so) {
    try {
      Ort::Session session(*ort_env.get(), buffer.data(), buffer.size(), so);
      ASSERT_FALSE(should_throw) << "Creation of session should have thrown";
    } catch (const std::exception& ex) {
      ASSERT_TRUE(should_throw) << "Creation of session should not have thrown. Exception:" << ex.what();
      ASSERT_THAT(ex.what(), testing::HasSubstr("ONNX format model is not supported in this build."));
    }
  };

  Ort::SessionOptions so;
  create_session(so);

#ifdef USE_CUDA
  // test with CUDA provider when using onnxruntime as dll
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(so, 0));
  create_session(so);
#endif
}

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD)
TEST(CApiTest, session_options_empty_affinity_string) {
  Ort::SessionOptions options;
  options.AddConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, "");
  constexpr auto model_path = ORT_TSTR("testdata/matmul_1.onnx");

  try {
    Ort::Session session(*ort_env.get(), model_path, options);
    ASSERT_TRUE(false) << "Creation of session should have thrown exception";
  } catch (const std::exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Affinity string must not be empty"));
  }
}
#endif

#endif

#ifdef DISABLE_EXTERNAL_INITIALIZERS
TEST(CApiTest, TestDisableExternalInitiliazers) {
  constexpr auto model_path = ORT_TSTR("testdata/model_with_external_initializers.onnx");

  Ort::SessionOptions so;
  try {
    Ort::Session session(*ort_env.get(), model_path, so);
    ASSERT_TRUE(false) << "Creation of session should have thrown exception";
  } catch (const std::exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Initializer tensors with external data is not allowed."));
  }
}

#elif !defined(ORT_MINIMAL_BUILD)
TEST(CApiTest, TestExternalInitializersInjection) {
  constexpr auto model_path = ORT_TSTR("testdata/model_with_external_initializer_come_from_user.onnx");
  std::array<int64_t, 4> Pads_not_on_disk{0, 0, 1, 1};
  constexpr std::array<int64_t, 1> init_shape{4};

  const std::vector<std::string> init_names{"Pads_not_on_disk"};
  std::vector<Ort::Value> initializer_data;

  auto cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto init_tensor = Ort::Value::CreateTensor(cpu_mem_info, Pads_not_on_disk.data(), Pads_not_on_disk.size(), init_shape.data(), init_shape.size());
  initializer_data.push_back(std::move(init_tensor));

  Ort::SessionOptions so;
  so.AddExternalInitializers(init_names, initializer_data);
  EXPECT_NO_THROW(Ort::Session(*ort_env, model_path, so));
}

#endif
}  // namespace test
}  // namespace onnxruntime
