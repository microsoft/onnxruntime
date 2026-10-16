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
//   - A factory returning nullptr causes the affected test to skip.
//   - Factory exceptions fail the test instead of hiding initialization regressions.
//
// Only documented, backend-agnostic contracts are asserted here. Memory that is
// not CPU-accessible is never dereferenced from the test thread; such checks are
// guarded by OrtDevice::UsesCpuMemory().

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "gtest/gtest-spi.h"

#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
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

// Reject unknown registered names even when the corresponding EP is not compiled in,
// as well as stale exemptions and entries that are both registered and exempted.
TEST(EpConformanceCoverage, RegistrationsAndExemptionsAreWellFormed) {
  const auto& all_ep_names = GetAllExecutionProviderNames();
  const auto params = GetEpConformanceParams();

  const auto is_known = [&](std::string_view ep_name) {
    return std::any_of(all_ep_names.begin(), all_ep_names.end(),
                       [ep_name](const std::string& name) { return std::string_view{name} == ep_name; });
  };

  for (const auto& param : params) {
    EXPECT_TRUE(is_known(param.ep_name))
        << param.ep_name << " is registered for EP conformance coverage but is not a known "
        << "execution provider name. Fix the typo or drop the stale entry.";
  }

  const auto check = [&](std::string_view exempt) {
    EXPECT_TRUE(is_known(exempt))
        << exempt << " is listed as exempt from EP conformance coverage but is not a known "
        << "execution provider name. Fix the typo or drop the stale entry.";

    EXPECT_FALSE(IsRegisteredForConformance(params, exempt))
        << exempt << " is both registered in GetEpConformanceParams() and listed as exempt. "
        << "Remove it from the exemption list.";
  };

  for (std::string_view ep_name : kStructurallyExemptEps) check(ep_name);
  for (std::string_view ep_name : kNotYetVettedEps) check(ep_name);
}

namespace {

class OrtEpConformanceProvider : public IExecutionProvider {
 public:
  explicit OrtEpConformanceProvider(const OrtEp& ort_ep)
      : IExecutionProvider("OrtEpConformanceProvider"), ort_ep_(ort_ep) {}

  const OrtEp* GetOrtEp() const override { return &ort_ep_; }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    ++registry_queries;
    return kernel_registry;
  }

  std::vector<AllocatorPtr> CreatePreferredAllocators() override {
    ++allocator_queries;
    return {};
  }

  std::shared_ptr<KernelRegistry> kernel_registry;
  mutable size_t registry_queries = 0;
  size_t allocator_queries = 0;

 private:
  const OrtEp& ort_ep_;
};

class OrtEpConformanceInvariantTest : public testing::Test {
 protected:
  OrtEpConformanceInvariantTest() : ep_(ort_ep_) {
    ort_ep_.ort_version_supported = ORT_API_VERSION;
    ort_ep_.GetCapability = GetCapability;
    ort_ep_.Compile = Compile;
    ort_ep_.ReleaseNodeComputeInfos = ReleaseNodeComputeInfos;
  }

  static OrtStatus* ORT_API_CALL GetCapability(OrtEp*, const OrtGraph*, OrtEpGraphSupportInfo*) noexcept {
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL Compile(OrtEp*, const OrtGraph**, const OrtNode**, size_t,
                                         OrtNodeComputeInfo**, OrtNode**) noexcept {
    return nullptr;
  }

  static void ORT_API_CALL ReleaseNodeComputeInfos(OrtEp*, OrtNodeComputeInfo**, size_t) noexcept {}

  static OrtStatus* ORT_API_CALL GetNullKernelRegistry(OrtEp*, const OrtKernelRegistry** registry) noexcept {
    *registry = nullptr;
    return nullptr;
  }

  OrtEp ort_ep_{};
  OrtEpConformanceProvider ep_;
};

}  // namespace

TEST_F(OrtEpConformanceInvariantTest, RequiredFunctionsArePresent) {
  ep_conformance::CheckOrtEpRequiredFunctionsArePresent(ep_, ep_.Type());
}

TEST_F(OrtEpConformanceInvariantTest, MissingGetCapabilityFails) {
  ort_ep_.GetCapability = nullptr;
  EXPECT_NONFATAL_FAILURE(ep_conformance::CheckOrtEpRequiredFunctionsArePresent(ep_, ep_.Type()),
                          "GetCapability must be implemented");
}

TEST_F(OrtEpConformanceInvariantTest, CompileBasedEpPasses) {
  ep_conformance::CheckOrtEpDeclaresCompileOrKernelRegistry(ep_, ep_.Type());
}

TEST_F(OrtEpConformanceInvariantTest, MissingCompileFails) {
  ort_ep_.Compile = nullptr;
  EXPECT_NONFATAL_FAILURE(ep_conformance::CheckOrtEpDeclaresCompileOrKernelRegistry(ep_, ep_.Type()),
                          "does not implement Compile()");
}

TEST_F(OrtEpConformanceInvariantTest, MissingReleaseNodeComputeInfosFails) {
  ort_ep_.ReleaseNodeComputeInfos = nullptr;
  EXPECT_NONFATAL_FAILURE(ep_conformance::CheckOrtEpDeclaresCompileOrKernelRegistry(ep_, ep_.Type()),
                          "must implement ReleaseNodeComputeInfos()");
}

TEST_F(OrtEpConformanceInvariantTest, NullRegistryCallbackRequiresCompile) {
  ort_ep_.GetKernelRegistry = GetNullKernelRegistry;
  ort_ep_.Compile = nullptr;
  EXPECT_NONFATAL_FAILURE(ep_conformance::CheckOrtEpDeclaresCompileOrKernelRegistry(ep_, ep_.Type()),
                          "does not implement Compile()");
}

TEST_F(OrtEpConformanceInvariantTest, NullRegistryCallbackRequiresReleaseNodeComputeInfos) {
  ort_ep_.GetKernelRegistry = GetNullKernelRegistry;
  ort_ep_.ReleaseNodeComputeInfos = nullptr;
  EXPECT_NONFATAL_FAILURE(ep_conformance::CheckOrtEpDeclaresCompileOrKernelRegistry(ep_, ep_.Type()),
                          "must implement ReleaseNodeComputeInfos()");
}

TEST_F(OrtEpConformanceInvariantTest, NullRegistryCallbackWithCompilePasses) {
  ort_ep_.GetKernelRegistry = GetNullKernelRegistry;
  ep_conformance::CheckOrtEpDeclaresCompileOrKernelRegistry(ep_, ep_.Type());
}

TEST_F(OrtEpConformanceInvariantTest, EmptyRegistryDoesNotRequireCompile) {
  // The adapter caches the registry returned by OrtEp, including an empty registry.
  ep_.kernel_registry = std::make_shared<KernelRegistry>();
  ort_ep_.Compile = nullptr;
  ort_ep_.ReleaseNodeComputeInfos = nullptr;
  ep_conformance::CheckOrtEpDeclaresCompileOrKernelRegistry(ep_, ep_.Type());
}

TEST_F(OrtEpConformanceInvariantTest, RegistryAndCompileMayCoexist) {
  ep_.kernel_registry = std::make_shared<KernelRegistry>();
  ep_conformance::CheckOrtEpDeclaresCompileOrKernelRegistry(ep_, ep_.Type());
}

TEST_F(OrtEpConformanceInvariantTest, Abi23DoesNotQueryRegistry) {
  ort_ep_.ort_version_supported = 23;
  ep_.kernel_registry = std::make_shared<KernelRegistry>();
  ort_ep_.Compile = nullptr;
  EXPECT_NONFATAL_FAILURE(ep_conformance::CheckOrtEpDeclaresCompileOrKernelRegistry(ep_, ep_.Type()),
                          "does not implement Compile()");
  EXPECT_EQ(ep_.registry_queries, 0u);
}

TEST_F(OrtEpConformanceInvariantTest, Abi22SkipsNewerFunctions) {
  ort_ep_.ort_version_supported = 22;
  ort_ep_.GetCapability = nullptr;
  ort_ep_.Compile = nullptr;
  ort_ep_.ReleaseNodeComputeInfos = nullptr;

  const auto checks = {ep_conformance::CheckOrtEpRequiredFunctionsArePresent,
                       ep_conformance::CheckOrtEpDeclaresCompileOrKernelRegistry};
  for (const auto check : checks) {
    testing::TestPartResultArray results;
    {
      testing::ScopedFakeTestPartResultReporter reporter(
          testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
      check(ep_, ep_.Type());
    }
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results.GetTestPartResult(0).type(), testing::TestPartResult::kSkip);
  }
  EXPECT_EQ(ep_.registry_queries, 0u);
}

TEST_F(OrtEpConformanceInvariantTest, PluginAllocatorChecksSkipBeforeQueryingAllocators) {
  const auto checks = {ep_conformance::CheckPreferredAllocatorsAreNonNullAndRepeatable,
                       ep_conformance::CheckPreferredAllocatorsAllocateUsableMemory,
                       ep_conformance::CheckDataTransferCpuCopyPreservesData,
                       ep_conformance::CheckPreferredAllocatorInfoIsConsistent};
  for (const auto check : checks) {
    testing::TestPartResultArray results;
    {
      testing::ScopedFakeTestPartResultReporter reporter(
          testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
      check(ep_, ep_.Type());
    }
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results.GetTestPartResult(0).type(), testing::TestPartResult::kSkip);
  }
  EXPECT_EQ(ep_.allocator_queries, 0u);
}

class EpConformanceTest : public testing::TestWithParam<EpConformanceParam> {
 protected:
  // A null result skips the test; exceptions propagate to gtest as test failures.
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
