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

// Testing helper function that runs a caller-provided QDQ graph (build_test_case) to allow the caller to
// 1) test which nodes are assigned to an EP, and 2) check that the inference output matches with the CPU EP.
void RunModelTest(const GetQDQTestCaseFn& build_test_case, const char* test_description,
                  const ProviderOptions& provider_options,
                  const EPVerificationParams& params = EPVerificationParams(),
                  const std::unordered_map<std::string, int>& domain_to_version = {});

enum HTPSupport {
  HTP_SUPPORT_UNKNOWN = 0,
  HTP_UNSUPPORTED,
  HTP_SUPPORTED,
  HTP_SUPPORT_ERROR,
};

// Testing helper function that calls QNN EP's GetCapability() function with a mock graph to check
// if the HTP backend is available.
// TODO: Remove once HTP can be emulated on Windows ARM64.
HTPSupport GetHTPSupport(const onnxruntime::logging::Logger& logger);

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