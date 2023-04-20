// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
TEST_F(QnnHTPBackendTests, TestQDQConvU8U8S32) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildQDQConvTestCase<uint8_t, uint8_t, int32_t, uint8_t>({1, 1, 5, 5},   // Input shape
                                                                           {1, 1, 3, 3}),  // Weights shape
                  provider_options,
                  18,                             // Opset
                  ExpectedEPNodeAssignment::All,  // All nodes assigned to QNN EP.
                  1,                              // Expect 1 fused node in graph.
                  "QDQConvU8U8S32");
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif