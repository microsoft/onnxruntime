// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS)

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/nn/conv_activation_fusion_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

TEST_P(ConvActivationFusionExecutionTests, ConvMulReluTanhPreservesBothActivations) {
  RunConvMulReluTanhFusionTest(GetParam(), [](SessionOptions& options) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  });
}

INSTANTIATE_TEST_SUITE_P(
    ExecutionProviders, ConvActivationFusionExecutionTests,
    ::testing::Values(ConvActivationFusionExecutionTestCase{
        "WebGPU", []() { return DefaultWebGpuExecutionProvider(); }, "Conv", kMSInternalNHWCDomain}),
    [](const ::testing::TestParamInfo<ConvActivationFusionExecutionTestCase>& info) { return info.param.name; });

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS)
