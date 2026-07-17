// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Execution Provider conformance test suite.
//
// These parameterized tests encode invariants that EVERY IExecutionProvider
// implementation is expected to satisfy, independent of the specific hardware
// backend. They turn the previously-implicit Liskov-substitutability
// assumptions of the IExecutionProvider contract into enforced, executable
// checks, so that a new (or modified) EP cannot silently violate the behavior
// the framework relies on.
//
// The invariant checks themselves live in
// test/util/include/ep_conformance_invariants.h and are shared with the plugin EP
// suite (test/providers/ep_conformance_plugin_test.cc), which runs the same
// invariants against a dynamically-loaded plugin EP.
//
// Adding an EP to the coverage is a single line: append an entry to
// GetEpConformanceParams() below, guarded by the appropriate USE_* macro. The
// stored value is a *factory*, not a constructed provider, so:
//   - No EP is instantiated during static initialization.
//   - A factory that returns nullptr (EP compiled but unavailable at runtime,
//     e.g. no GPU present) causes the affected test to be skipped, not failed.
//
// Only documented, backend-agnostic contracts are asserted here. Memory that is
// not CPU-accessible is never dereferenced from the test thread; such checks are
// guarded by OrtDevice::UsesCpuMemory().

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/execution_provider.h"

#include "test/util/include/default_providers.h"
#include "test/util/include/ep_conformance_invariants.h"

namespace onnxruntime {
namespace test {

namespace {

// One EP under test: a human-readable name (also used as the gtest parameter
// suffix, so it must be a valid identifier) plus a factory that constructs a
// fresh provider instance.
struct EpConformanceParam {
  std::string name;
  std::function<std::unique_ptr<IExecutionProvider>()> factory;
  bool expects_plugin_ep = false;
};

std::vector<EpConformanceParam> GetEpConformanceParams() {
  std::vector<EpConformanceParam> params;

  // CPU is always available. Cover both the arena and non-arena allocator paths
  // since they are distinct IAllocator implementations with different Alloc/Free
  // behavior.
  params.push_back({"Cpu_Arena", [] { return DefaultCpuExecutionProvider(/*enable_arena*/ true); }});
  params.push_back({"Cpu_NoArena", [] { return DefaultCpuExecutionProvider(/*enable_arena*/ false); }});

#ifdef USE_CUDA
#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP) && defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE)
  params.push_back({"Cuda", [] { return DefaultCudaExecutionProvider(); }, true});
#else
  params.push_back({"Cuda", [] { return DefaultCudaExecutionProvider(); }});
#endif
#endif
#ifdef USE_DML
  params.push_back({"Dml", [] { return DefaultDmlExecutionProvider(); }});
#endif
  // Mirror the guard used by base_tester.cc / default_providers.cc: in
  // ORT_USE_EP_API_ADAPTERS builds DefaultWebGpuExecutionProvider() ORT_ENFORCEs
  // (aborting the whole test run) when the dynamic plugin EP is initialized to a
  // different EP, rather than cleanly returning nullptr. Only list the built-in
  // WebGPU EP when it is not routed through the EP API adapters.
#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
  params.push_back({"WebGpu", [] { return DefaultWebGpuExecutionProvider(); }});
#endif
#ifdef USE_XNNPACK
  params.push_back({"Xnnpack", [] { return DefaultXnnpackExecutionProvider(); }});
#endif

  return params;
}

}  // namespace

class EpConformanceTest : public testing::TestWithParam<EpConformanceParam> {
 protected:
  // Construct the EP under test. Returns nullptr when the EP is compiled but not
  // available in the current environment; callers should GTEST_SKIP() in that case.
  std::unique_ptr<IExecutionProvider> MakeEp() const { return GetParam().factory(); }
};

// Invariant: Type() is non-empty and stable -- both across repeated calls on a
// single instance and across independent instances from the same factory. The
// framework keys kernel registries and node assignment on this string.
TEST_P(EpConformanceTest, TypeIsNonEmptyAndStable) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckTypeIsNonEmptyAndStable(*ep, [this] { return MakeEp(); }, GetParam().name);
}

// Invariant: GetPreferredLayout() returns one of the defined DataLayout values.
// Layout transformation dispatches on this, so an out-of-range value is a bug.
TEST_P(EpConformanceTest, PreferredLayoutIsValid) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckPreferredLayoutIsValid(*ep, GetParam().name);
}

// Invariant: the CPU mem types always map to CPU-accessible memory. The
// framework's input/output staging copies depend on this for every EP.
TEST_P(EpConformanceTest, CpuMemTypesMapToCpuAccessibleDevice) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckCpuMemTypesMapToCpuAccessibleDevice(*ep, GetParam().name);
}

// Invariant: CreatePreferredAllocators() never yields a null allocator and is
// repeatable. The header documents it as a stateless factory, so a second call
// must produce an equivalently-sized set.
TEST_P(EpConformanceTest, PreferredAllocatorsAreNonNullAndRepeatable) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckPreferredAllocatorsAreNonNullAndRepeatable(*ep, GetParam().name);
}

// Invariant: each CPU-accessible preferred allocator hands back usable memory:
// a non-zero allocation yields a non-null, host-writable and -readable pointer
// that can be freed. Device allocators are intentionally excluded here -- their
// raw Alloc/Free lifecycle is backend-specific (see body) -- and are covered by
// PreferredAllocatorsAreNonNullAndRepeatable instead.
TEST_P(EpConformanceTest, PreferredAllocatorsAllocateUsableMemory) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckPreferredAllocatorsAllocateUsableMemory(*ep, GetParam().name);
}

// Invariant: GetDataTransfer() is optional (may be null). When provided, and it
// advertises the ability to copy within a CPU-accessible device, a CopyTensor
// round-trip must preserve the data exactly.
TEST_P(EpConformanceTest, DataTransferCpuRoundTripPreservesData) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckDataTransferCpuRoundTripPreservesData(*ep, GetParam().name);
}

// Invariant: read-only metadata queries are callable and self-consistent on a
// freshly constructed EP (no session or logger required), and the device-id
// accessor agrees with GetDevice().
TEST_P(EpConformanceTest, MetadataQueriesAreCallable) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckMetadataQueriesAreCallable(*ep, GetParam().name);
}

// Invariant: GetGraphCaptureNodeAssignmentPolicy() returns one of the defined
// OrtGraphCaptureNodeAssignmentPolicy values. The session dispatches on this
// while validating a graph for capture, so an out-of-range value is a bug.
// This is a pure query and is valid to call on every EP regardless of whether
// graph capture is enabled.
TEST_P(EpConformanceTest, GraphCaptureNodeAssignmentPolicyIsValid) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckGraphCaptureNodeAssignmentPolicyIsValid(*ep, GetParam().name);
}

// Invariant: a built-in EP returns no backing OrtEp. A PluginExecutionProvider
// returns the same non-null backing OrtEp across repeated queries.
TEST_P(EpConformanceTest, GetOrtEpMatchesProviderKind) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckGetOrtEpMatchesProviderKind(*ep, GetParam().expects_plugin_ep, GetParam().name);
}

// Invariant: GetEpContextNodes() reports no nodes on a freshly constructed EP.
// EPs populate this only when generating an EPContext cache model during
// compilation; with no compilation performed, the documented default is empty.
TEST_P(EpConformanceTest, EpContextNodesEmptyOnFreshEp) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckEpContextNodesEmptyOnFreshEp(*ep, GetParam().name);
}

// Invariant: every preferred allocator reports self-consistent OrtMemoryInfo --
// a non-empty name and a valid allocator type. This metadata keys allocator
// lookup in the framework, so it must be well-formed for every EP. Only the
// backend-agnostic fields are checked; the raw memory is not touched here.
TEST_P(EpConformanceTest, PreferredAllocatorInfoIsConsistent) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckPreferredAllocatorInfoIsConsistent(*ep, GetParam().name);
}

INSTANTIATE_TEST_SUITE_P(
    EpContract, EpConformanceTest, testing::ValuesIn(GetEpConformanceParams()),
    [](const testing::TestParamInfo<EpConformanceParam>& info) { return info.param.name; });

}  // namespace test
}  // namespace onnxruntime
