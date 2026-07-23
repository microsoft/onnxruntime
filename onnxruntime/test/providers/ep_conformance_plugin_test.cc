// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Plugin EP conformance test suite.
//
// This suite runs the shared Execution Provider conformance invariants
// (test/util/include/ep_conformance_invariants.h) against a *plugin* EP -- an
// OrtEp reached through the PluginExecutionProvider wrapper -- rather than against
// an in-tree IExecutionProvider implementation. Developing an EP as a plugin (the
// OrtEp / OrtEpFactory ABI) is the recommended path for external EP authors, so the
// same Liskov-substitutability contract that the built-in suite
// (test/framework/execution_provider_conformance_test.cc) enforces for the internal
// interface is enforced here for the plugin interface.
//
// The EP under test is whichever plugin EP the dynamic plugin EP infrastructure was
// initialized with. That is intentionally open-ended so the suite covers arbitrary
// plugin EPs:
//   - the in-repo example plugin EP,
//   - an ORT-owned EP that has been converted to a plugin (e.g. CUDA or WebGPU
//     routed through the EP API adapters), or
//   - any plugin EP specified at runtime via the
//     ORT_UNIT_TEST_MAIN_DYNAMIC_PLUGIN_EP_CONFIG_JSON[_FILE] environment variables.
//
// When the infrastructure is not initialized (no plugin EP configured for this run),
// MakeEp() returns nullptr and every test cleanly skips.

#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "core/framework/execution_provider.h"

#include "test/unittest_util/test_dynamic_plugin_ep.h"
#include "test/util/include/ep_conformance_invariants.h"

#if defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE)

namespace onnxruntime {
namespace test {
namespace {

namespace dynamic_plugin_ep_infra = onnxruntime::test::dynamic_plugin_ep_infra;

// Fixture that constructs the dynamically-loaded plugin EP under test. A null return
// means the dynamic plugin EP infrastructure was not initialized for this run, in
// which case the test skips. Each test is the plugin-interface counterpart of the
// identically-named test in the built-in EP suite.
class EpPluginConformanceTest : public ::testing::Test {
 protected:
  static std::unique_ptr<IExecutionProvider> MakeEp() { return dynamic_plugin_ep_infra::MakeEp(); }

  // Human-readable label for failure messages: the selected plugin EP's name.
  static std::string EpLabel() { return dynamic_plugin_ep_infra::GetEpName().value_or(std::string{"PluginEp"}); }
};

TEST_F(EpPluginConformanceTest, TypeIsNonEmptyAndStable) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << "dynamic plugin EP infrastructure is not initialized.";

  ep_conformance::CheckTypeIsNonEmptyAndStable(
      *ep, [] { return EpPluginConformanceTest::MakeEp(); }, EpLabel());
}

TEST_F(EpPluginConformanceTest, PreferredLayoutIsValid) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << "dynamic plugin EP infrastructure is not initialized.";

  ep_conformance::CheckPreferredLayoutIsValid(*ep, EpLabel());
}

TEST_F(EpPluginConformanceTest, CpuMemTypesMapToCpuAccessibleDevice) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << "dynamic plugin EP infrastructure is not initialized.";

  ep_conformance::CheckCpuMemTypesMapToCpuAccessibleDevice(*ep, EpLabel());
}

TEST_F(EpPluginConformanceTest, PreferredAllocatorsAreNonNullAndRepeatable) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << "dynamic plugin EP infrastructure is not initialized.";

  ep_conformance::CheckPreferredAllocatorsAreNonNullAndRepeatable(*ep, EpLabel());
}

TEST_F(EpPluginConformanceTest, PreferredAllocatorsAllocateUsableMemory) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << "dynamic plugin EP infrastructure is not initialized.";

  ep_conformance::CheckPreferredAllocatorsAllocateUsableMemory(*ep, EpLabel());
}

TEST_F(EpPluginConformanceTest, DataTransferCpuRoundTripPreservesData) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << "dynamic plugin EP infrastructure is not initialized.";

  ep_conformance::CheckDataTransferCpuRoundTripPreservesData(*ep, EpLabel());
}

TEST_F(EpPluginConformanceTest, MetadataQueriesAreCallable) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << "dynamic plugin EP infrastructure is not initialized.";

  ep_conformance::CheckMetadataQueriesAreCallable(*ep, EpLabel());
}

TEST_F(EpPluginConformanceTest, GraphCaptureNodeAssignmentPolicyIsValid) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << "dynamic plugin EP infrastructure is not initialized.";

  ep_conformance::CheckGraphCaptureNodeAssignmentPolicyIsValid(*ep, EpLabel());
}

TEST_F(EpPluginConformanceTest, GetOrtEpMatchesProviderKind) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << "dynamic plugin EP infrastructure is not initialized.";

  // A dynamically-loaded plugin EP is always backed by an OrtEp, so unlike a
  // built-in EP it must report a stable non-null backing OrtEp.
  ep_conformance::CheckGetOrtEpMatchesProviderKind(*ep, /*expects_plugin_ep*/ true, EpLabel());
}

TEST_F(EpPluginConformanceTest, EpContextNodesEmptyOnFreshEp) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << "dynamic plugin EP infrastructure is not initialized.";

  ep_conformance::CheckEpContextNodesEmptyOnFreshEp(*ep, EpLabel());
}

TEST_F(EpPluginConformanceTest, PreferredAllocatorInfoIsConsistent) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << "dynamic plugin EP infrastructure is not initialized.";

  ep_conformance::CheckPreferredAllocatorInfoIsConsistent(*ep, EpLabel());
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE)
