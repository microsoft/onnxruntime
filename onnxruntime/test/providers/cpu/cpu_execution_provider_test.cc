// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_registry.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {
TEST(CPUExecutionProviderTest, MetadataTest) {
  CPUExecutionProviderInfo info;
  auto provider = std::make_unique<CPUExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);
  ASSERT_EQ(provider->GetOrtDeviceByMemType(OrtMemTypeDefault).Type(), OrtDevice::CPU);
}

TEST(CPUExecutionProviderTest, Float16GemmAndMatMulRegistration) {
  auto kernel_registry = CPUExecutionProvider(CPUExecutionProviderInfo()).GetKernelRegistry();
  ASSERT_NE(kernel_registry, nullptr);

  const auto has_kernel = [&](std::string_view op_type, int opset_version, const DataTypeImpl* tensor_type) {
    KernelRegistry::TypeConstraintMap type_constraints{{"T", tensor_type}};
    const KernelCreateInfo* kernel_create_info{};
    const auto status = kernel_registry->TryFindKernel(
        kCpuExecutionProvider, op_type, kOnnxDomain, opset_version, type_constraints,
        DefaultLoggingManager().DefaultLogger(), &kernel_create_info);
    return status.IsOK() && kernel_create_info != nullptr;
  };

  const auto* fp32_tensor_type = DataTypeImpl::GetTensorType<float>();
  EXPECT_TRUE(has_kernel("Gemm", 10, fp32_tensor_type));
  EXPECT_TRUE(has_kernel("Gemm", 12, fp32_tensor_type));
  EXPECT_TRUE(has_kernel("MatMul", 10, fp32_tensor_type));
  EXPECT_TRUE(has_kernel("MatMul", 12, fp32_tensor_type));

  const auto* fp16_tensor_type = DataTypeImpl::GetTensorType<MLFloat16>();
  const bool expected_kernel = MlasHalfGemmAccelerationSupported(nullptr);
  EXPECT_EQ(has_kernel("Gemm", 8, fp16_tensor_type), expected_kernel);
  EXPECT_EQ(has_kernel("Gemm", 10, fp16_tensor_type), expected_kernel);
  EXPECT_EQ(has_kernel("Gemm", 12, fp16_tensor_type), expected_kernel);
  EXPECT_EQ(has_kernel("Gemm", 13, fp16_tensor_type), expected_kernel);
  EXPECT_EQ(has_kernel("MatMul", 8, fp16_tensor_type), expected_kernel);
  EXPECT_EQ(has_kernel("MatMul", 10, fp16_tensor_type), expected_kernel);
  EXPECT_EQ(has_kernel("MatMul", 12, fp16_tensor_type), expected_kernel);
  EXPECT_EQ(has_kernel("MatMul", 13, fp16_tensor_type), expected_kernel);
}

TEST(CPUExecutionProviderTest, Float16MatMulRunsOnCpu) {
  OpTester test("MatMul", 13);
  test.AddInput<MLFloat16>("A", {2, 2},
                           {MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f), MLFloat16(4.0f)});
  test.AddInput<MLFloat16>("B", {2, 2},
                           {MLFloat16(5.0f), MLFloat16(6.0f), MLFloat16(7.0f), MLFloat16(8.0f)});
  test.AddOutput<MLFloat16>("Y", {2, 2},
                            {MLFloat16(19.0f), MLFloat16(22.0f), MLFloat16(43.0f), MLFloat16(50.0f)});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.ConfigEps(std::move(execution_providers)).RunWithConfig();
}

TEST(CPUExecutionProviderTest, Float16GemmRunsOnCpu) {
  OpTester test("Gemm", 13);
  test.AddInput<MLFloat16>("A", {2, 2},
                           {MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f), MLFloat16(4.0f)});
  test.AddInput<MLFloat16>("B", {2, 2},
                           {MLFloat16(5.0f), MLFloat16(6.0f), MLFloat16(7.0f), MLFloat16(8.0f)});
  test.AddInput<MLFloat16>("C", {2, 2},
                           {MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f), MLFloat16(4.0f)});
  test.AddOutput<MLFloat16>("Y", {2, 2},
                            {MLFloat16(20.0f), MLFloat16(24.0f), MLFloat16(46.0f), MLFloat16(54.0f)});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.ConfigEps(std::move(execution_providers)).RunWithConfig();
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
