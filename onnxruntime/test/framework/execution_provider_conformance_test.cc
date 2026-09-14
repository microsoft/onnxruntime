// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Execution Provider conformance test suite.
//
// These parameterized tests encode invariants that every IExecutionProvider
// implementation is expected to satisfy, independent of the specific hardware
// backend. They turn the previously-implicit Liskov-substitutability
// assumptions of the IExecutionProvider contract into enforced, executable
// checks, so that a covered EP cannot silently violate the behavior the
// framework relies on.
//
// Coverage is enforced, not opt-in: EpConformanceCoverage.EveryAvailableEpIsRegistered
// cross-checks the registered list below against GetAvailableExecutionProviderNames(),
// the registry of EPs compiled into this build. An EP that is compiled but neither
// registered nor explicitly exempted fails that test, so coverage cannot silently
// regress as EPs are added.
//
// The invariant checks themselves live in
// test/util/include/ep_conformance_invariants.h and are shared with the plugin EP
// suite (test/providers/ep_conformance_plugin_test.cc), which runs the same
// invariants against a dynamically-loaded plugin EP.
//
// Adding an EP to the coverage is a single line: append an entry to
// GetEpConformanceParams() below. No USE_* guard is required -- every
// Default*ExecutionProvider() is declared unconditionally and returns nullptr when
// its EP is not compiled in. The stored value is a *factory*, not a constructed
// provider, so:
//   - No EP is instantiated during static initialization.
//   - An EP that is compiled but unavailable at runtime (e.g. no GPU present)
//     causes the affected test to be skipped, not failed -- whether its factory
//     signals that by returning nullptr or by throwing during construction (see
//     MakeEp()).
//
// Only documented, backend-agnostic contracts are asserted here. Memory that is
// not CPU-accessible is never dereferenced from the test thread; such checks are
// guarded by OrtDevice::UsesCpuMemory().

#include <algorithm>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
#include "core/providers/get_execution_providers.h"

#include "test/util/include/default_providers.h"
#include "test/util/include/ep_conformance_invariants.h"

namespace onnxruntime {
namespace test {

namespace {

// One EP under test:
//   - name: a human-readable label, also used as the gtest parameter suffix, so it
//     must be a valid identifier. Several entries may share an ep_name when one EP is
//     covered in more than one configuration (e.g. CPU with and without the arena).
//   - ep_name: the canonical provider name (kXxxExecutionProvider), used to
//     cross-check this list against the EPs compiled into the build.
//   - factory: constructs a fresh provider instance (see MakeEp()).
//   - expects_plugin_ep: true iff this EP is plugin-backed, i.e. GetOrtEp() must
//     return non-null (see CheckGetOrtEpMatchesProviderKind). Built-in EPs leave it
//     false.
struct EpConformanceParam {
  std::string name;
  std::string_view ep_name;
  std::function<std::unique_ptr<IExecutionProvider>()> factory;
  bool expects_plugin_ep = false;
};

std::vector<EpConformanceParam> GetEpConformanceParams() {
  std::vector<EpConformanceParam> params;

  // CPU is always available. Cover both the arena and non-arena allocator paths
  // since they are distinct IAllocator implementations with different Alloc/Free
  // behavior.
  params.push_back({"Cpu_Arena", kCpuExecutionProvider,
                    [] { return DefaultCpuExecutionProvider(/*enable_arena*/ true); }});
  params.push_back({"Cpu_NoArena", kCpuExecutionProvider,
                    [] { return DefaultCpuExecutionProvider(/*enable_arena*/ false); }});

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP) && defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE)
  params.push_back({"Cuda", kCudaExecutionProvider, [] { return DefaultCudaExecutionProvider(); },
                    /*expects_plugin_ep=*/true});
#else
  params.push_back({"Cuda", kCudaExecutionProvider, [] { return DefaultCudaExecutionProvider(); }});
#endif

  params.push_back({"Dml", kDmlExecutionProvider, [] { return DefaultDmlExecutionProvider(); }});

  // Mirror the guard used by base_tester.cc / default_providers.cc: in
  // ORT_USE_EP_API_ADAPTERS builds DefaultWebGpuExecutionProvider() ORT_ENFORCEs
  // (aborting the whole test run) when the dynamic plugin EP is initialized to a
  // different EP, rather than cleanly returning nullptr. Only list the built-in
  // WebGPU EP when it is not routed through the EP API adapters. This matches the
  // guard on kWebGpuExecutionProvider in get_execution_providers.cc, so the coverage
  // cross-check below stays consistent in both configurations.
#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
  params.push_back({"WebGpu", kWebGpuExecutionProvider, [] { return DefaultWebGpuExecutionProvider(); }});
#endif

  params.push_back({"Xnnpack", kXnnpackExecutionProvider, [] { return DefaultXnnpackExecutionProvider(); }});

  return params;
}

// EPs that are compiled into some builds but deliberately not covered above. The two
// reasons are kept in separate lists so the difference stays visible to reviewers.

// (1) Not conformance-testable from a native gtest binary. These have no
//     Default*ExecutionProvider() helper and are not expected to grow one.
constexpr std::string_view kStructurallyExemptEps[] = {
    kJsExecutionProvider,       // web/emscripten builds only
    kWebNNExecutionProvider,    // web/emscripten builds only
    kAzureExecutionProvider,    // remote inference endpoint, not a local compute EP
    kVitisAIExecutionProvider,  // requires external configuration/runtime to construct
};

// (2) Not yet vetted. Each of these does have a Default*ExecutionProvider() and is
//     expected to graduate into GetEpConformanceParams() above. They are parked here
//     rather than registered so that introducing this cross-check does not start
//     running eleven never-before-exercised invariants across many CI legs at once.
//     An EP should be moved out of this list in the same change that validates it and
//     fixes whatever the invariants surface for it.
constexpr std::string_view kNotYetVettedEps[] = {
    kAclExecutionProvider,
    kCannExecutionProvider,
    kCoreMLExecutionProvider,
    kDnnlExecutionProvider,
    kMIGraphXExecutionProvider,
    kNnapiExecutionProvider,
    kNvTensorRTRTXExecutionProvider,
    kOpenVINOExecutionProvider,
    kQnnExecutionProvider,
    kRknpuExecutionProvider,
    kTensorrtExecutionProvider,
    kVSINPUExecutionProvider,
};

bool IsExemptFromConformanceCoverage(std::string_view ep_name) {
  for (std::string_view exempt : kStructurallyExemptEps) {
    if (exempt == ep_name) return true;
  }
  for (std::string_view exempt : kNotYetVettedEps) {
    if (exempt == ep_name) return true;
  }
  return false;
}

bool IsRegisteredForConformance(const std::vector<EpConformanceParam>& params, std::string_view ep_name) {
  return std::any_of(params.begin(), params.end(),
                     [ep_name](const EpConformanceParam& param) { return param.ep_name == ep_name; });
}

}  // namespace

// Meta-test: every EP compiled into this build is either exercised by the suite below
// or explicitly exempted. This is what keeps the conformance guarantee from quietly
// decaying -- adding a new EP to the build fails here until someone either registers
// it in GetEpConformanceParams() or records why it cannot be covered.
//
// The check is deliberately one-directional: it does not assert the converse (that
// every registered EP is "available"), because the two lists legitimately differ that
// way. DefaultSnpeExecutionProvider() exists, for instance, while SNPE has no entry in
// the availability registry at all.
TEST(EpConformanceCoverage, EveryAvailableEpIsRegistered) {
  const auto params = GetEpConformanceParams();

  for (const std::string& available : GetAvailableExecutionProviderNames()) {
    const std::string_view ep_name{available};
    if (IsExemptFromConformanceCoverage(ep_name)) continue;

    EXPECT_TRUE(IsRegisteredForConformance(params, ep_name))
        << available << " is compiled into this build but has no EP conformance coverage. "
        << "Register it in GetEpConformanceParams(), or add it to kStructurallyExemptEps "
        << "or kNotYetVettedEps (in this file) with the reason.";
  }
}

// Guards the exemption lists themselves. Without this, a typo or a stale entry left
// behind after an EP is renamed or removed would silently widen the exemption and hide
// a real coverage gap -- the exact failure mode the check above exists to prevent.
TEST(EpConformanceCoverage, ExemptionsAreWellFormed) {
  const auto& all_ep_names = GetAllExecutionProviderNames();
  const auto params = GetEpConformanceParams();

  const auto check = [&](std::string_view exempt) {
    EXPECT_TRUE(std::any_of(all_ep_names.begin(), all_ep_names.end(),
                            [exempt](const std::string& name) { return std::string_view{name} == exempt; }))
        << exempt << " is listed as exempt from EP conformance coverage but is not a known "
        << "execution provider name. Fix the typo or drop the stale entry.";

    EXPECT_FALSE(IsRegisteredForConformance(params, exempt))
        << exempt << " is both registered in GetEpConformanceParams() and listed as exempt. "
        << "Remove it from the exemption list.";
  };

  for (std::string_view ep_name : kStructurallyExemptEps) check(ep_name);
  for (std::string_view ep_name : kNotYetVettedEps) check(ep_name);
}

class EpConformanceTest : public testing::TestWithParam<EpConformanceParam> {
 protected:
  // Construct the EP under test. Returns nullptr when the EP is compiled but not
  // available in the current environment; callers should GTEST_SKIP() in that case.
  // Some providers signal unavailability by returning nullptr from their factory;
  // others (e.g. CUDA, whose constructor calls cudaSetDevice) throw when no device or
  // driver is present. Both are treated as "unavailable" so the affected test skips
  // rather than fails; the exception text is logged so a genuine construction
  // regression stays visible.
  std::unique_ptr<IExecutionProvider> MakeEp() const {
    try {
      return GetParam().factory();
    } catch (const std::exception& e) {
      GTEST_LOG_(WARNING) << GetParam().name
                          << " EP factory threw during construction (treated as unavailable): " << e.what();
      return nullptr;
    }
  }
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
// advertises the ability to copy within a CPU-accessible device, a CPU-to-CPU
// CopyTensor must preserve the data exactly.
TEST_P(EpConformanceTest, DataTransferCpuCopyPreservesData) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckDataTransferCpuCopyPreservesData(*ep, GetParam().name);
}

// Invariant: read-only metadata queries are callable on a freshly constructed EP (no
// session or logger required). GetDeviceId() is provider-defined and is not required
// to equal GetDevice().Id(), so it is only smoke-called.
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

// Invariant: an OrtEp implements the functions ORT dereferences without a null check.
// Skips for a built-in EP, which has no backing OrtEp.
TEST_P(EpConformanceTest, OrtEpRequiredFunctionsArePresent) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckOrtEpRequiredFunctionsArePresent(*ep, GetParam().name);
}

// Invariant: an OrtEp declares a coherent execution mode -- it exposes a kernel
// registry, or it implements both Compile() and ReleaseNodeComputeInfos(). Skips for a
// built-in EP, which has no backing OrtEp.
TEST_P(EpConformanceTest, OrtEpDeclaresCompileOrKernelRegistry) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckOrtEpDeclaresCompileOrKernelRegistry(*ep, GetParam().name);
}

// Invariant: GetEpContextNodes() reports no nodes on a freshly constructed EP.
// EPs populate this only when generating an EPContext cache model during
// compilation; with no compilation performed, the documented default is empty.
TEST_P(EpConformanceTest, EpContextNodesEmptyOnFreshEp) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  ep_conformance::CheckEpContextNodesEmptyOnFreshEp(*ep, GetParam().name);
}

// Invariant: every preferred allocator reports a valid allocator type in its
// OrtMemoryInfo. This metadata keys allocator lookup in the framework, so it must
// be well-formed for every EP. The allocator name is intentionally not asserted --
// an empty OrtMemoryInfo.name is permitted by the contract. Only the
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
