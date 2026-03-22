// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
TEST(CPUExecutionProviderTest, MetadataTest) {
  CPUExecutionProviderInfo info;
  auto provider = std::make_unique<CPUExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);
  ASSERT_EQ(provider->GetOrtDeviceByMemType(OrtMemTypeDefault).Type(), OrtDevice::CPU);
}

TEST(CPUExecutionProviderTest, MlasBackendKernelSelectorDefaultsToKleidiAiEnabled) {
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  ConfigOptions config_options;

  SetupMlasBackendKernelSelectorFromConfigOptions(config, config_options);

  EXPECT_TRUE(config.use_kleidiai);
  EXPECT_FALSE(config.use_nchwc_pointwise_filter_repacking);
}

TEST(CPUExecutionProviderTest, MlasBackendKernelSelectorCanDisableKleidiAi) {
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  ConfigOptions config_options;
  const Status add_config_status = config_options.AddConfigEntry(kOrtSessionOptionsMlasDisableKleidiAi, "1");
  ASSERT_TRUE(add_config_status.IsOK()) << add_config_status.ErrorMessage();

  SetupMlasBackendKernelSelectorFromConfigOptions(config, config_options);

  EXPECT_FALSE(config.use_kleidiai);
}

TEST(CPUExecutionProviderTest, MlasBackendKernelSelectorCanEnableNchwcPointwiseFilterRepacking) {
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  ConfigOptions config_options;
  const Status add_config_status = config_options.AddConfigEntry(
      kOrtSessionOptionsMlasEnableNchwcPointwiseFilterRepacking, "1");
  ASSERT_TRUE(add_config_status.IsOK()) << add_config_status.ErrorMessage();

  SetupMlasBackendKernelSelectorFromConfigOptions(config, config_options);

  EXPECT_TRUE(config.use_kleidiai);
  EXPECT_TRUE(config.use_nchwc_pointwise_filter_repacking);
}
}  // namespace test
}  // namespace onnxruntime
