// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/config_options.h"
#include "core/framework/provider_options.h"
#include "core/graph/constants.h"
#include "core/session/ep_cache_versioning.h"
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
  std::string long_affinity_str(ConfigOptions::kMaxValueLength + 1, '0');
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
    FAIL() << "Appending CUDA options have thrown exception";
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

// ============================================================================
// EP Cache Versioning Tests
// ============================================================================

namespace {

#ifdef ORT_VERSION
std::string GetVersionedCachePath(const std::string& base_path) {
  if (base_path.empty()) return base_path;
  return base_path + "/" + ORT_VERSION;
}
#endif

// Ensure EP cache path option keys are registered for tests that call the
// generic versioning helpers directly with provider names.
const bool kRegisterTestEpCachePathOptions = []() {
  RegisterEpCachePathOptions("CoreML", {"ModelCacheDirectory"});
  RegisterEpCachePathOptions("TensorRT", {"trt_engine_cache_path", "trt_timing_cache_path"});
  RegisterEpCachePathOptions("MIGraphX", {"migraphx_model_cache_dir"});
  RegisterEpCachePathOptions("NvTensorRtRtx", {"nv_runtime_cache_path"});
  return true;
}();

}  // namespace

TEST(CApiTest, ep_cache_versioning_config_options_disabled) {
  ConfigOptions config;
  (void)config.AddConfigEntry(kOrtSessionOptionsEpCacheUseOrtVersion, "0");
  (void)config.AddConfigEntry("ep.coreml.ModelCacheDirectory", "/tmp/coreml_cache");
  (void)config.AddConfigEntry("ep.tensorrt.trt_engine_cache_path", "/tmp/trt_cache");

  ApplyEpCacheVersionToConfigOptions(config);

  EXPECT_EQ(config.GetConfigOrDefault("ep.coreml.ModelCacheDirectory", ""), "/tmp/coreml_cache");
  EXPECT_EQ(config.GetConfigOrDefault("ep.tensorrt.trt_engine_cache_path", ""), "/tmp/trt_cache");
}

TEST(CApiTest, ep_cache_versioning_config_options_enabled) {
  ConfigOptions config;
  (void)config.AddConfigEntry(kOrtSessionOptionsEpCacheUseOrtVersion, "1");
  (void)config.AddConfigEntry("ep.coreml.ModelCacheDirectory", "/tmp/coreml_cache");
  (void)config.AddConfigEntry("ep.tensorrt.trt_engine_cache_path", "/tmp/trt_cache");
  (void)config.AddConfigEntry("ep.migraphx.migraphx_model_cache_dir", "/tmp/migraphx");
  (void)config.AddConfigEntry("ep.nvtensorrtrtx.nv_runtime_cache_path", "/tmp/nv_cache");

  ApplyEpCacheVersionToConfigOptions(config);

#ifdef ORT_VERSION
  EXPECT_EQ(config.GetConfigOrDefault("ep.coreml.ModelCacheDirectory", ""), GetVersionedCachePath("/tmp/coreml_cache"));
  EXPECT_EQ(config.GetConfigOrDefault("ep.tensorrt.trt_engine_cache_path", ""), GetVersionedCachePath("/tmp/trt_cache"));
  EXPECT_EQ(config.GetConfigOrDefault("ep.migraphx.migraphx_model_cache_dir", ""), GetVersionedCachePath("/tmp/migraphx"));
  EXPECT_EQ(config.GetConfigOrDefault("ep.nvtensorrtrtx.nv_runtime_cache_path", ""), GetVersionedCachePath("/tmp/nv_cache"));
#endif
}

TEST(CApiTest, ep_cache_versioning_config_options_unknown_provider) {
  ConfigOptions config;
  (void)config.AddConfigEntry(kOrtSessionOptionsEpCacheUseOrtVersion, "1");
  (void)config.AddConfigEntry("ep.unknown_ep.cache_path", "/tmp/unknown");
  (void)config.AddConfigEntry("ep.custom_provider.cache_dir", "/tmp/custom");

  ApplyEpCacheVersionToConfigOptions(config);

  EXPECT_EQ(config.GetConfigOrDefault("ep.unknown_ep.cache_path", ""), "/tmp/unknown");
  EXPECT_EQ(config.GetConfigOrDefault("ep.custom_provider.cache_dir", ""), "/tmp/custom");
}

TEST(CApiTest, ep_cache_versioning_config_options_empty_path) {
  ConfigOptions config;
  (void)config.AddConfigEntry(kOrtSessionOptionsEpCacheUseOrtVersion, "1");
  (void)config.AddConfigEntry("ep.coreml.ModelCacheDirectory", "");

  ApplyEpCacheVersionToConfigOptions(config);

  EXPECT_EQ(config.GetConfigOrDefault("ep.coreml.ModelCacheDirectory", ""), "");
}

TEST(CApiTest, ep_cache_versioning_config_options_non_cache_keys) {
  ConfigOptions config;
  (void)config.AddConfigEntry(kOrtSessionOptionsEpCacheUseOrtVersion, "1");
  (void)config.AddConfigEntry("ep.coreml.EnableOnSubgraph", "1");
  (void)config.AddConfigEntry("ep.tensorrt.trt_max_workspace_size", "1073741824");

  ApplyEpCacheVersionToConfigOptions(config);

  EXPECT_EQ(config.GetConfigOrDefault("ep.coreml.EnableOnSubgraph", ""), "1");
  EXPECT_EQ(config.GetConfigOrDefault("ep.tensorrt.trt_max_workspace_size", ""), "1073741824");
}

TEST(CApiTest, ep_cache_versioning_provider_options_disabled) {
  ConfigOptions config;
  (void)config.AddConfigEntry(kOrtSessionOptionsEpCacheUseOrtVersion, "0");

  ProviderOptions opts;
  opts["ModelCacheDirectory"] = "/tmp/coreml";

  auto result = GetProviderOptionsWithVersionedCachePaths(opts, config, "CoreML");

  EXPECT_EQ(result["ModelCacheDirectory"], "/tmp/coreml");
}

TEST(CApiTest, ep_cache_versioning_provider_options_enabled) {
  ConfigOptions config;
  (void)config.AddConfigEntry(kOrtSessionOptionsEpCacheUseOrtVersion, "1");

  {
    ProviderOptions opts;
    opts["ModelCacheDirectory"] = "/tmp/coreml";
    auto result = GetProviderOptionsWithVersionedCachePaths(opts, config, "CoreML");
#ifdef ORT_VERSION
    EXPECT_EQ(result["ModelCacheDirectory"], GetVersionedCachePath("/tmp/coreml"));
#endif
  }

  {
    ProviderOptions opts;
    opts["trt_engine_cache_path"] = "/tmp/trt_engine";
    opts["trt_timing_cache_path"] = "/tmp/trt_timing";
    auto result = GetProviderOptionsWithVersionedCachePaths(opts, config, "TensorRT");
#ifdef ORT_VERSION
    EXPECT_EQ(result["trt_engine_cache_path"], GetVersionedCachePath("/tmp/trt_engine"));
    EXPECT_EQ(result["trt_timing_cache_path"], GetVersionedCachePath("/tmp/trt_timing"));
#endif
  }

  {
    ProviderOptions opts;
    opts["migraphx_model_cache_dir"] = "/tmp/migraphx";
    auto result = GetProviderOptionsWithVersionedCachePaths(opts, config, "MIGraphX");
#ifdef ORT_VERSION
    EXPECT_EQ(result["migraphx_model_cache_dir"], GetVersionedCachePath("/tmp/migraphx"));
#endif
  }

  {
    ProviderOptions opts;
    opts["nv_runtime_cache_path"] = "/tmp/nv";
    auto result = GetProviderOptionsWithVersionedCachePaths(opts, config, "NvTensorRTRTX");
#ifdef ORT_VERSION
    EXPECT_EQ(result["nv_runtime_cache_path"], GetVersionedCachePath("/tmp/nv"));
#endif
  }
}

TEST(CApiTest, ep_cache_versioning_provider_options_case_insensitive) {
  ConfigOptions config;
  (void)config.AddConfigEntry(kOrtSessionOptionsEpCacheUseOrtVersion, "1");

  ProviderOptions opts;
  opts["ModelCacheDirectory"] = "/tmp/coreml_case";

  auto result = GetProviderOptionsWithVersionedCachePaths(opts, config, "CoReMl");

#ifdef ORT_VERSION
  EXPECT_EQ(result["ModelCacheDirectory"], GetVersionedCachePath("/tmp/coreml_case"));
#endif
}

TEST(CApiTest, ep_cache_versioning_provider_options_unknown_provider) {
  ConfigOptions config;
  (void)config.AddConfigEntry(kOrtSessionOptionsEpCacheUseOrtVersion, "1");

  ProviderOptions opts;
  opts["some_cache_key"] = "/tmp/unknown_provider";

  auto result = GetProviderOptionsWithVersionedCachePaths(opts, config, "UnknownEP");

  EXPECT_EQ(result["some_cache_key"], "/tmp/unknown_provider");
}

TEST(CApiTest, ep_cache_versioning_provider_options_empty_path) {
  ConfigOptions config;
  (void)config.AddConfigEntry(kOrtSessionOptionsEpCacheUseOrtVersion, "1");

  ProviderOptions opts;
  opts["ModelCacheDirectory"] = "";

  auto result = GetProviderOptionsWithVersionedCachePaths(opts, config, "CoreML");

  EXPECT_EQ(result["ModelCacheDirectory"], "");
}

TEST(CApiTest, ep_cache_versioning_provider_options_non_cache_keys) {
  ConfigOptions config;
  (void)config.AddConfigEntry(kOrtSessionOptionsEpCacheUseOrtVersion, "1");

  ProviderOptions opts;
  opts["EnableOnSubgraph"] = "1";
  opts["MaxComputeUnits"] = "2";
  opts["some_other_option"] = "/tmp/not_a_cache";

  auto result = GetProviderOptionsWithVersionedCachePaths(opts, config, "CoreML");

  EXPECT_EQ(result["EnableOnSubgraph"], "1");
  EXPECT_EQ(result["MaxComputeUnits"], "2");
  EXPECT_EQ(result["some_other_option"], "/tmp/not_a_cache");
}

TEST(CApiTest, ep_cache_versioning_provider_options_does_not_modify_input) {
  ConfigOptions config;
  (void)config.AddConfigEntry(kOrtSessionOptionsEpCacheUseOrtVersion, "1");

  ProviderOptions opts;
  opts["ModelCacheDirectory"] = "/tmp/coreml";
  ProviderOptions original = opts;

  auto result = GetProviderOptionsWithVersionedCachePaths(opts, config, "CoreML");

  EXPECT_EQ(opts["ModelCacheDirectory"], original["ModelCacheDirectory"]);

#ifdef ORT_VERSION
  EXPECT_EQ(result["ModelCacheDirectory"], GetVersionedCachePath("/tmp/coreml"));
#endif
}
