// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ORT_WEBGPU_INTERNAL_TEST_LIBRARY_PATH)

#include "gtest/gtest.h"

#include "core/providers/providers.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "test/providers/webgpu/test_plugin.h"
#include "test/unittest_util/test_dynamic_plugin_ep.h"
#include "test/util/include/asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime::test {

TEST(WebGpuTestPluginOptionsTest, DefaultsAndPerCallOverrides) {
  dynamic_plugin_ep_infra::RunWithTemporaryShutdownForTesting([]() {
    EXPECT_TRUE(dynamic_plugin_ep_infra::GetDefaultEpOptions().empty());
    for (bool default_nchw : {false, true}) {
      dynamic_plugin_ep_infra::InitializationConfig init;
      init.ep_library_registration_name = "WebGpuTestPluginOptions";
      init.selected_ep_name = kWebGpuExecutionProvider;
      init.default_ep_options = {{"preferredLayout", default_nchw ? "NCHW" : "NHWC"}};
      ASSERT_STATUS_OK(dynamic_plugin_ep_infra::Initialize(*ort_env, init));
      auto shutdown = gsl::finally([]() { dynamic_plugin_ep_infra::Shutdown(); });

      EXPECT_EQ(GetWebGpuTestPlugin().MakeEp()->GetPreferredLayout(),
                default_nchw ? DataLayout::NCHW : DataLayout::NHWC);
      ConfigOptions overrides;
      ASSERT_STATUS_OK(overrides.AddConfigEntry(webgpu::options::kPreferredLayout, default_nchw ? "NHWC" : "NCHW"));
      EXPECT_EQ(GetWebGpuTestPlugin().MakeEp(overrides)->GetPreferredLayout(),
                default_nchw ? DataLayout::NHWC : DataLayout::NCHW);
      EXPECT_EQ(dynamic_plugin_ep_infra::GetDefaultEpOptions(), init.default_ep_options);
    }
  });
}

TEST(WebGpuTestPluginOptionsTest, ContextUsesConfiguredDefaults) {
  dynamic_plugin_ep_infra::RunWithTemporaryShutdownForTesting([]() {
    dynamic_plugin_ep_infra::InitializationConfig init;
    init.ep_library_registration_name = "WebGpuTestPluginOptions";
    init.selected_ep_name = kWebGpuExecutionProvider;
    init.default_ep_options = {{"dawnBackendType", "invalid-test-backend"}};
    ASSERT_STATUS_OK(dynamic_plugin_ep_infra::Initialize(*ort_env, init));
    auto shutdown = gsl::finally([]() { dynamic_plugin_ep_infra::Shutdown(); });

    try {
      auto ep = GetWebGpuTestPlugin().MakeEp();
      FAIL() << "The test context ignored the invalid configured backend";
    } catch (const Ort::Exception& error) {
      EXPECT_NE(std::string{error.what()}.find("Invalid Dawn backend type: invalid-test-backend"), std::string::npos);
    }
  });
}

}  // namespace onnxruntime::test

#endif  // defined(ORT_WEBGPU_INTERNAL_TEST_LIBRARY_PATH)
