// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include "core/framework/execution_provider.h"
#include "test/unittest_util/test_dynamic_plugin_ep.h"

#include <limits>
#include <string>

#include "test/util/include/asserts.h"

#if defined(USE_CUDA) && defined(ORT_USE_EP_API_ADAPTERS)
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/providers/cuda/plugin/cuda_kernel_adapter.h"
#endif

namespace onnxruntime::test {

namespace dynamic_plugin_ep_test_infra = onnxruntime::test::dynamic_plugin_ep_infra;

TEST(DynamicPluginEpInfraTest, ParseInitializationConfigParsesOptionalFields) {
  constexpr std::string_view kConfigJson = R"json(
{
  "ep_library_registration_name": "CudaPluginExecutionProvider",
  "ep_library_path": "/tmp/libonnxruntime_providers_cuda_plugin.so",
  "selected_ep_device_indices": [0, 2],
  "default_ep_options": {
    "ep.cuda.use_tf32": "1",
    "ep.cuda.prefer_nhwc_layout": "1"
  },
  "tests_to_skip": [
    "CudaTests.SkipMe",
    "GraphTests.SkipMeToo"
  ]
}
)json";

  dynamic_plugin_ep_test_infra::InitializationConfig config{};
  ASSERT_STATUS_OK(dynamic_plugin_ep_test_infra::ParseInitializationConfig(kConfigJson, config));

  EXPECT_EQ(config.ep_library_registration_name, "CudaPluginExecutionProvider");
  EXPECT_EQ(config.ep_library_path, "/tmp/libonnxruntime_providers_cuda_plugin.so");
  EXPECT_TRUE(config.selected_ep_name.empty());
  EXPECT_THAT(config.selected_ep_device_indices, ::testing::ElementsAre(0u, 2u));
  EXPECT_THAT(config.default_ep_options,
              ::testing::UnorderedElementsAre(
                  ::testing::Pair("ep.cuda.prefer_nhwc_layout", "1"),
                  ::testing::Pair("ep.cuda.use_tf32", "1")));
  EXPECT_THAT(config.tests_to_skip,
              ::testing::ElementsAre("CudaTests.SkipMe", "GraphTests.SkipMeToo"));
}

TEST(DynamicPluginEpInfraTest, ParseInitializationConfigDefaultsUnsetOptionalFields) {
  constexpr std::string_view kConfigJson = R"json(
{
  "ep_library_registration_name": "ExamplePluginEP",
  "ep_library_path": "/tmp/libexample_plugin_ep.so",
  "selected_ep_name": "ExampleExecutionProvider"
}
)json";

  dynamic_plugin_ep_test_infra::InitializationConfig config{};
  ASSERT_STATUS_OK(dynamic_plugin_ep_test_infra::ParseInitializationConfig(kConfigJson, config));

  EXPECT_EQ(config.ep_library_registration_name, "ExamplePluginEP");
  EXPECT_EQ(config.ep_library_path, "/tmp/libexample_plugin_ep.so");
  EXPECT_EQ(config.selected_ep_name, "ExampleExecutionProvider");
  EXPECT_TRUE(config.selected_ep_device_indices.empty());
  EXPECT_TRUE(config.default_ep_options.empty());
  EXPECT_TRUE(config.tests_to_skip.empty());
}

TEST(DynamicPluginEpInfraTest, ParseInitializationConfigRejectsMissingRequiredFields) {
  constexpr std::string_view kConfigJson = R"json(
{
  "ep_library_registration_name": "CudaPluginExecutionProvider"
}
)json";

  dynamic_plugin_ep_test_infra::InitializationConfig config{};
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(dynamic_plugin_ep_test_infra::ParseInitializationConfig(kConfigJson, config),
                                      "JSON parse error");
}

TEST(DynamicPluginEpInfraTest, UninitializedStateReturnsSafeDefaults) {
  dynamic_plugin_ep_test_infra::Shutdown();

  EXPECT_FALSE(dynamic_plugin_ep_test_infra::IsInitialized());
  EXPECT_EQ(dynamic_plugin_ep_test_infra::MakeEp(), nullptr);
  EXPECT_FALSE(dynamic_plugin_ep_test_infra::GetEpName().has_value());
  EXPECT_TRUE(dynamic_plugin_ep_test_infra::GetTestsToSkip().empty());

  dynamic_plugin_ep_test_infra::Shutdown();

  EXPECT_FALSE(dynamic_plugin_ep_test_infra::IsInitialized());
  EXPECT_FALSE(dynamic_plugin_ep_test_infra::GetEpName().has_value());
  EXPECT_TRUE(dynamic_plugin_ep_test_infra::GetTestsToSkip().empty());
}

#if defined(USE_CUDA) && defined(ORT_USE_EP_API_ADAPTERS)
TEST(DynamicPluginEpInfraTest, CudaKernelAdapterRuntimeConfigExposesFuseConvBiasAndSdpaKernel) {
  onnxruntime::CUDAExecutionProvider provider{"CudaPluginExecutionProvider"};
  auto& config = onnxruntime::cuda::detail::GetCudaKernelAdapterRuntimeConfigForProvider(&provider);
  config.fuse_conv_bias = true;
  config.sdpa_kernel = static_cast<int>(onnxruntime::contrib::attention::AttentionBackend::MATH);

  EXPECT_TRUE(provider.IsFuseConvBias());

  const auto* attention_kernel_options = provider.GetAttentionKernelOptions();
  EXPECT_TRUE(attention_kernel_options->UseUnfusedAttention());
  EXPECT_FALSE(attention_kernel_options->UseFlashAttention());
  EXPECT_FALSE(attention_kernel_options->UseEfficientAttention());
  EXPECT_FALSE(attention_kernel_options->UseCudnnFlashAttention());
}

TEST(DynamicPluginEpInfraTest, CudaKernelAdapterTryBytesForCountDetectsOverflow) {
  size_t bytes = 0;
  EXPECT_FALSE(onnxruntime::cuda::detail::TryBytesForCount(std::numeric_limits<size_t>::max(), 2, bytes));
}

TEST(DynamicPluginEpInfraTest, CudaKernelAdapterTryBytesForCountPreservesRawByteCounts) {
  size_t bytes = 0;
  ASSERT_TRUE(onnxruntime::cuda::detail::TryBytesForCount(123, 0, bytes));
  EXPECT_EQ(bytes, size_t{123});
}

TEST(DynamicPluginEpInfraTest, CudaKernelAdapterTryBytesForCountNormalCase) {
  size_t bytes = 0;
  ASSERT_TRUE(onnxruntime::cuda::detail::TryBytesForCount(10, 4, bytes));
  EXPECT_EQ(bytes, size_t{40});
}
#endif

}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD) && defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
