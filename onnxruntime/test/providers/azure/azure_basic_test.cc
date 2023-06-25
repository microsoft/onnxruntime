// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_allocator.h"
#include "gtest/gtest.h"

// defined in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

TEST(AzureEP, TestSessionCreation) {
  const auto* ort_model_path = ORT_TSTR("testdata/mul_1.onnx");
  Ort::SessionOptions so;
  so.AddConfigEntry("azure.endpoint_type", "triton");
  onnxruntime::ProviderOptions options;
  so.AppendExecutionProvider("AZURE", options);
  // session could be created
  EXPECT_NO_THROW((Ort::Session{*ort_env, ort_model_path, so}));
}

}  // namespace test
}  // namespace onnxruntime