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
}

TEST(CPUExecutionProviderTest, MlasBackendKernelSelectorCanDisableKleidiAi) {
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  ConfigOptions config_options;
  const Status add_config_status = config_options.AddConfigEntry(kOrtSessionOptionsMlasDisableKleidiAi, "1");
  ASSERT_TRUE(add_config_status.IsOK()) << add_config_status.ErrorMessage();

  SetupMlasBackendKernelSelectorFromConfigOptions(config, config_options);

  EXPECT_FALSE(config.use_kleidiai);
}

TEST(CPUExecutionProviderTest, MlasBackendKernelSelectorParsesKleidiAiConvIgemmMaxWork) {
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  ConfigOptions config_options;
  const Status add_config_status = config_options.AddConfigEntry(kOrtSessionOptionsMlasKleidiAiConvIgemmMaxWork, "1234567");
  ASSERT_TRUE(add_config_status.IsOK()) << add_config_status.ErrorMessage();

  SetupMlasBackendKernelSelectorFromConfigOptions(config, config_options);

  EXPECT_EQ(config.kleidiai_conv_igemm_max_work, 1234567u);
}

TEST(CPUExecutionProviderTest, MlasBackendKernelSelectorRejectsInvalidKleidiAiConvIgemmMaxWork) {
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  ConfigOptions config_options;
  const Status add_config_status = config_options.AddConfigEntry(kOrtSessionOptionsMlasKleidiAiConvIgemmMaxWork, "Not a Number");
  ASSERT_TRUE(add_config_status.IsOK()) << add_config_status.ErrorMessage();

  try {
    SetupMlasBackendKernelSelectorFromConfigOptions(config, config_options);
    FAIL() << "Expected invalid " << kOrtSessionOptionsMlasKleidiAiConvIgemmMaxWork << " to throw.";
  } catch (const OnnxRuntimeException& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find(kOrtSessionOptionsMlasKleidiAiConvIgemmMaxWork), std::string::npos);
    EXPECT_NE(message.find("Not a Number"), std::string::npos);
    EXPECT_NE(message.find("Expected a non-negative integer."), std::string::npos);
  }
}

}  // namespace test
}  // namespace onnxruntime
