// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include <string>
#include <unordered_map>
#include "core/framework/provider_options.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/util/include/test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

using GetTestModelFn = std::function<void(ModelTestBuilder& builder)>;

/**
 * Runs a test model on the QNN EP. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param build_test_case Function that builds a test model. See test/optimizer/qdq_test_utils.h
 * \param provider_options Provider options for QNN EP.
 * \param opset_version The opset version.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 * \param num_modes_in_ep The expected number of nodes assigned to QNN EP's partition.
 * \param test_description Description of the test for error reporting.
 */
void RunQnnModelTest(const GetTestModelFn& build_test_case, const ProviderOptions& provider_options,
                     int opset_version, ExpectedEPNodeAssignment expected_ep_assignment, int num_nodes_in_ep,
                     const char* test_description);

enum HTPSupport {
  HTP_SUPPORT_UNKNOWN = 0,
  HTP_UNSUPPORTED,
  HTP_SUPPORTED,
  HTP_SUPPORT_ERROR,
};

// Testing fixture class for tests that require the HTP backend. Checks if HTP is available before the test begins.
// The test is skipped if HTP is unavailable (may occur on Windows ARM64).
// TODO: Remove once HTP can be emulated on Windows ARM64.
class QnnHTPBackendTests : public ::testing::Test {
 protected:
  void SetUp() override;

  static HTPSupport cached_htp_support_;  // Set by the first test using this fixture.
};

/**
 * Returns true if the given reduce operator type (e.g., "ReduceSum") and opset version (e.g., 13)
 * supports "axes" as an input (instead of an attribute).
 *
 * \param op_type The string denoting the reduce operator's type (e.g., "ReduceSum").
 * \param opset_version The opset of the operator.
 *
 * \return True if "axes" is an input, or false if "axes" is an attribute.
 */
bool ReduceOpHasAxesInput(const std::string& op_type, int opset_version);

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)