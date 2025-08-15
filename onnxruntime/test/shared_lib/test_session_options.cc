// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gmock/gmock.h"

using namespace onnxruntime;

TEST(CApiTest, session_options_graph_optimization_level) {
  // Test set optimization level succeeds when valid level is provided.
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
}

TEST(CApiTest, session_options_deterministic_compute) {
  // Manual validation currently. Check that SetDeterministicCompute in abi_session_options.cc is hit.
  Ort::SessionOptions options;
  options.SetDeterministicCompute(true);
}

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD) && !defined(ORT_NO_EXCEPTIONS)

TEST(CApiTest, session_options_oversized_affinity_string) {
  Ort::SessionOptions options;
  std::string long_affinity_str(onnxruntime::kMaxStrLen + 1, '0');
  try {
    options.AddConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, long_affinity_str.c_str());
    ASSERT_TRUE(false) << "Creation of config should have thrown exception";
  } catch (const std::exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Config value is longer than maximum length: "));
  }
}

#endif

#if defined(USE_OPENVINO_PROVIDER_INTERFACE)
// Test that loading OpenVINO EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_add_openvino) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  Ort::Status status = Ort::Status{api.SessionOptionsAppendExecutionProvider(session_options,
                                                                             kOpenVINOExecutionProvider,
                                                                             nullptr, nullptr, 0)};
  ASSERT_TRUE(!status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("Failed to load"));
}
#endif  // defined(USE_OPENVINO_PROVIDER_INTERFACE)

#if defined(USE_CUDA_PROVIDER_INTERFACE)
// Test that loading CUDA EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_add_cuda) {
  Ort::SessionOptions session_options;

  Ort::CUDAProviderOptions cuda_options;
  try {
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    ASSERT_TRUE(false) << "Appending CUDA options have thrown exception";
  } catch (const Ort::Exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Failed to load"));
  }
}
#endif  // defined(USE_CUDA_PROVIDER_INTERFACE)

#if defined(USE_NV_PROVIDER_INTERFACE)
// Test that loading NV EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_add_nv) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  Ort::Status status = Ort::Status{api.SessionOptionsAppendExecutionProvider(session_options,
                                                                             kNvTensorRTRTXExecutionProvider,
                                                                             nullptr, nullptr, 0)};
  ASSERT_TRUE(!status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("Failed to load"));
}
#endif  // defined(USE_OPENVINO_PROVIDER_INTERFACE)

#if defined(USE_TENSORRT_PROVIDER_INTERFACE)
// Test that loading TensorRT EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_add_tensorrt) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  OrtTensorRTProviderOptionsV2* trt_options = nullptr;
  Ort::Status status1 = Ort::Status{api.CreateTensorRTProviderOptions(&trt_options)};
  ASSERT_TRUE(status1.IsOK());

  Ort::Status status2 = Ort::Status{api.SessionOptionsAppendExecutionProvider_TensorRT_V2(session_options,
                                                                                          trt_options)};
  ASSERT_FALSE(status2.IsOK());
  EXPECT_EQ(status2.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status2.GetErrorMessage(), testing::HasSubstr("Failed to load"));

  api.ReleaseTensorRTProviderOptions(trt_options);
}
#endif  // defined(USE_TENSORRT_PROVIDER_INTERFACE)

#if defined(USE_VITISAI_PROVIDER_INTERFACE)
// Test that loading VitisAI EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_vitisai) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  Ort::Status status = Ort::Status{api.SessionOptionsAppendExecutionProvider(session_options,
                                                                             kVitisAIExecutionProvider,
                                                                             nullptr, nullptr, 0)};
  ASSERT_TRUE(!status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("Failed to load"));
}
#endif  // defined(USE_VITISAI_PROVIDER_INTERFACE)

#if defined(USE_QNN_PROVIDER_INTERFACE)
// Test that loading QNN EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_qnn) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  Ort::Status status = Ort::Status{api.SessionOptionsAppendExecutionProvider(session_options,
                                                                             kQnnExecutionProvider,
                                                                             nullptr, nullptr, 0)};
  ASSERT_TRUE(!status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("Failed to load"));
}
#endif  // defined(USE_QNN_PROVIDER_INTERFACE)
