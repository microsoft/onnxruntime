// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/inference_session.h"
#include "core/framework/session_options.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

#include "ssr/qnn_mock_ssr_controller.h"

namespace onnxruntime {
namespace test {

#if defined(_WIN32) && (defined(_M_ARM64) || defined(_M_ARM64EC))

class QnnMockSSRBackendTests : public QnnHTPBackendTests {
 protected:
  void SetUp() override;
  void TearDown() override;
#if defined(_WIN32)
  HMODULE lib_handle;
  FARPROC addr;
#endif  // defined(_WIN32)
  QnnMockSSRController* controller = nullptr;
  TestInputDef<float> input_def;
  TestInputDef<float> scale_def;
  TestInputDef<float> bias_def;
  ProviderOptions provider_options;
};

void QnnMockSSRBackendTests::SetUp() {
#if defined(_WIN32) && (defined(_M_ARM64) || defined(_M_ARM64EC))
#include <windows.h>
  QnnHTPBackendTests::SetUp();
  lib_handle = LoadLibraryW(L"QnnMockSSR.dll");
  ASSERT_NE(lib_handle, nullptr) << "Failed to load QnnMockSSR.dll";

  typedef QnnMockSSRController* (*GetQnnMockSSRControllerFn_t)();
  GetQnnMockSSRControllerFn_t GetQnnMockSSRController = reinterpret_cast<GetQnnMockSSRControllerFn_t>(
      GetProcAddress(lib_handle, "GetQnnMockSSRController"));
  ASSERT_NE(GetQnnMockSSRController, nullptr) << "Failed to get GetQnnMockSSRController function";

  controller = GetQnnMockSSRController();
  ASSERT_NE(controller, nullptr) << "GetQnnMockSSRController returned null";

#endif  // defined(_WIN32) && (defined(_M_ARM64) || defined(_M_ARM64EC))
  input_def = TestInputDef<float>({1, 2, 3, 3}, false, {-10.0f, 10.0f});
  scale_def = TestInputDef<float>({2}, true, {1.0f, 2.0f});
  bias_def = TestInputDef<float>({2}, true, {1.0f, 3.0f});
  provider_options = {
      {"backend_path", "QnnMockSSR.dll"},
      {"offload_graph_io_quantization", "0"},
      {"enable_ssr_handling", "1"},
  };
}

void QnnMockSSRBackendTests::TearDown() {
#if defined(_WIN32) && (defined(_M_ARM64) || defined(_M_ARM64EC))
  if (lib_handle) {
    FreeLibrary(lib_handle);
  }
#endif  // defined(_WIN32) && (defined(_M_ARM64) || defined(_M_ARM64EC))
}

TEST_F(QnnMockSSRBackendTests, SSRBackendGetBuildId) {
  controller->SetTiming(QnnMockSSRController::Timing::BackendGetBuildId);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}

TEST_F(QnnMockSSRBackendTests, SSRBackendCreate) {
  controller->SetTiming(QnnMockSSRController::Timing::BackendCreate);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}

TEST_F(QnnMockSSRBackendTests, SSRContextCreate) {
  controller->SetTiming(QnnMockSSRController::Timing::ContextCreate);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}

TEST_F(QnnMockSSRBackendTests, DISABLED_SSRBackendValidateOpConfig) {
  controller->SetTiming(QnnMockSSRController::Timing::BackendValidateOpConfig);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}

TEST_F(QnnMockSSRBackendTests, SSRLogCreate) {
  controller->SetTiming(QnnMockSSRController::Timing::LogCreate);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}

TEST_F(QnnMockSSRBackendTests, SSRGraphCreate) {
  controller->SetTiming(QnnMockSSRController::Timing::GraphCreate);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}

TEST_F(QnnMockSSRBackendTests, SSRGraphRetrieve) {
  controller->SetTiming(QnnMockSSRController::Timing::GraphRetrieve);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}

TEST_F(QnnMockSSRBackendTests, SSRContextGetBinarySize) {
  controller->SetTiming(QnnMockSSRController::Timing::ContextGetBinarySize);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}

TEST_F(QnnMockSSRBackendTests, SSRContextGetBinary) {
  controller->SetTiming(QnnMockSSRController::Timing::ContextGetBinary);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}

TEST_F(QnnMockSSRBackendTests, SSRTensorCreateGraphTensor) {
  controller->SetTiming(QnnMockSSRController::Timing::TensorCreateGraphTensor);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}

TEST_F(QnnMockSSRBackendTests, SSRGraphAddNode) {
  controller->SetTiming(QnnMockSSRController::Timing::GraphAddNode);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}

TEST_F(QnnMockSSRBackendTests, SSRGraphFinalize) {
  controller->SetTiming(QnnMockSSRController::Timing::GraphFinalize);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}

TEST_F(QnnMockSSRBackendTests, SSRGraphExecute) {
  controller->SetTiming(QnnMockSSRController::Timing::GraphExecute);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64)
}  // namespace test
}  // namespace onnxruntime

#endif
