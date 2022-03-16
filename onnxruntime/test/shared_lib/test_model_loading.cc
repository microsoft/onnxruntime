// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"
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
    buffer.resize(file.tellg());
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

TEST(CApiTest, TestModelWithExternalDataFromArray) {
  const char* model_path = "testdata/model_with_external_initializers.onnx";
  const char* external_data_path = "testdata/Pads.bin";
  const char* external_data_name = "Pads.bin";

  auto read_buffer = [](const char* file_path) {
    std::string buffer;
    std::ifstream file(file_path, std::ios::binary);
    if (!file)
      ORT_THROW("Error reading model");
    std::ostringstream oss;
    oss << file.rdbuf();
    buffer = oss.str();
    file.close();
    return buffer;
  };

  std::string model_buffer = read_buffer(model_path), external_data_buffer = read_buffer(external_data_path);

#if (!ORT_MINIMAL_BUILD)
  bool should_throw = false;
#else
  bool should_throw = true;
#endif

  std::vector<std::string> external_data_names = {external_data_name};
  std::vector<void*> external_data_buffers = {external_data_buffer.data()};

  auto create_session = [&](Ort::SessionOptions& so) {
    try {
      Ort::Session session(*ort_env.get(),
                           model_buffer.data(),
                           model_buffer.size(),
                           external_data_names.data(),
                           external_data_buffers.data(),
                           external_data_buffers.size(),
                           so);
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
#endif

#ifdef DISABLE_EXTERNAL_INITIALIZERS
TEST(CApiTest, TestDisableExternalInitiliazers) {

  const char* model_path = "testdata/model_with_external_initializers.onnx";

  Ort::SessionOptions so;
  try {
    Ort::Session session(*ort_env.get(), model_path, so);
    ASSERT_TRUE(false) << "Creation of session should have thrown exception";
  } catch (const std::exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Initializer tensors with external data is not allowed."));
  }
}
#endif
}  // namespace test
}  // namespace onnxruntime
