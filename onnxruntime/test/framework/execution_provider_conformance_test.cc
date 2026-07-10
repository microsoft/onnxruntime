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

#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/allocator.h"
#include "core/framework/data_transfer.h"
#include "core/framework/data_types.h"
#include "core/framework/execution_provider.h"
#include "core/framework/tensor.h"

#include "test/util/include/default_providers.h"

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

  const std::string type = ep->Type();
  EXPECT_FALSE(type.empty()) << "IExecutionProvider::Type() must not be empty.";
  EXPECT_EQ(type, ep->Type()) << "Type() must be stable across calls on the same instance.";

  auto ep2 = MakeEp();
  ASSERT_NE(ep2, nullptr);
  EXPECT_EQ(type, ep2->Type()) << "Type() must be identical for instances from the same factory.";
}

// Invariant: GetPreferredLayout() returns one of the defined DataLayout values.
// Layout transformation dispatches on this, so an out-of-range value is a bug.
TEST_P(EpConformanceTest, PreferredLayoutIsValid) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  const DataLayout layout = ep->GetPreferredLayout();
  EXPECT_TRUE(layout == DataLayout::NCHW || layout == DataLayout::NHWC)
      << "GetPreferredLayout() returned an unknown DataLayout value.";
}

// Invariant: the CPU mem types always map to CPU-accessible memory. The
// framework's input/output staging copies depend on this for every EP.
TEST_P(EpConformanceTest, CpuMemTypesMapToCpuAccessibleDevice) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  EXPECT_TRUE(ep->GetOrtDeviceByMemType(OrtMemTypeCPUInput).UsesCpuMemory())
      << "OrtMemTypeCPUInput must map to CPU-accessible memory.";
  EXPECT_TRUE(ep->GetOrtDeviceByMemType(OrtMemTypeCPUOutput).UsesCpuMemory())
      << "OrtMemTypeCPUOutput must map to CPU-accessible memory.";
}

// Invariant: CreatePreferredAllocators() never yields a null allocator and is
// repeatable. The header documents it as a stateless factory, so a second call
// must produce an equivalently-sized set.
TEST_P(EpConformanceTest, PreferredAllocatorsAreNonNullAndRepeatable) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  auto allocators = ep->CreatePreferredAllocators();
  for (const auto& alloc : allocators) {
    EXPECT_NE(alloc, nullptr) << "CreatePreferredAllocators() must not return null entries.";
  }

  auto allocators2 = ep->CreatePreferredAllocators();
  EXPECT_EQ(allocators.size(), allocators2.size())
      << "CreatePreferredAllocators() must be repeatable (documented as stateless).";
}

// Invariant: each CPU-accessible preferred allocator hands back usable memory:
// a non-zero allocation yields a non-null, host-writable and -readable pointer
// that can be freed. Device allocators are intentionally excluded here -- their
// raw Alloc/Free lifecycle is backend-specific (see body) -- and are covered by
// PreferredAllocatorsAreNonNullAndRepeatable instead.
TEST_P(EpConformanceTest, PreferredAllocatorsAllocateUsableMemory) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  auto allocators = ep->CreatePreferredAllocators();
  if (allocators.empty()) {
    GTEST_SKIP() << GetParam().name << " EP exposes no preferred allocators.";
  }

  constexpr size_t kBytes = 256;
  size_t exercised = 0;
  for (const auto& alloc : allocators) {
    ASSERT_NE(alloc, nullptr);

    // Standalone Alloc()/Free() is only a backend-agnostic contract for
    // CPU-accessible allocators. A device allocator may hand out memory with a
    // backend-specific lifecycle that this test cannot drive: e.g. the WebGPU
    // GpuBufferAllocator creates buffers mapped at creation that must be
    // unmapped through the buffer manager before Free(), so Free()-ing a
    // freshly allocated buffer throws. Skip such allocators; they are covered
    // by PreferredAllocatorsAreNonNullAndRepeatable.
    if (!alloc->Info().device.UsesCpuMemory()) {
      continue;
    }

    void* p = alloc->Alloc(kBytes);
    ASSERT_NE(p, nullptr) << "Alloc(" << kBytes << ") returned null for allocator on "
                          << alloc->Info().device.ToString();

    std::memset(p, 0xAB, kBytes);
    const auto* bytes = static_cast<const unsigned char*>(p);
    EXPECT_EQ(bytes[0], 0xAB);
    EXPECT_EQ(bytes[kBytes - 1], 0xAB);
    alloc->Free(p);
    ++exercised;
  }

  if (exercised == 0) {
    GTEST_SKIP() << GetParam().name
                 << " EP exposes no CPU-accessible preferred allocator to exercise.";
  }
}

// Invariant: GetDataTransfer() is optional (may be null). When provided, and it
// advertises the ability to copy within a CPU-accessible device, a CopyTensor
// round-trip must preserve the data exactly.
TEST_P(EpConformanceTest, DataTransferCpuRoundTripPreservesData) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  auto data_transfer = ep->GetDataTransfer();
  if (!data_transfer) {
    GTEST_SKIP() << GetParam().name << " EP provides no IDataTransfer (allowed by the contract).";
  }

  // Host the tensors on a CPU-accessible preferred allocator.
  AllocatorPtr cpu_alloc;
  for (auto& alloc : ep->CreatePreferredAllocators()) {
    if (alloc && alloc->Info().device.UsesCpuMemory()) {
      cpu_alloc = alloc;
      break;
    }
  }
  if (!cpu_alloc) {
    GTEST_SKIP() << GetParam().name << " EP has no CPU-accessible allocator to drive the round-trip.";
  }

  const OrtDevice& cpu_device = cpu_alloc->Info().device;
  if (!data_transfer->CanCopy(cpu_device, cpu_device)) {
    GTEST_SKIP() << GetParam().name << " EP data transfer does not advertise CPU<->CPU copy.";
  }

  const TensorShape shape({2, 3});
  Tensor src(DataTypeImpl::GetType<float>(), shape, cpu_alloc);
  Tensor dst(DataTypeImpl::GetType<float>(), shape, cpu_alloc);

  const float values[] = {1.f, -2.f, 3.5f, 4.f, 5.f, -6.25f};
  constexpr size_t kNumValues = sizeof(values) / sizeof(values[0]);
  ASSERT_EQ(static_cast<size_t>(shape.Size()), kNumValues);
  std::memcpy(src.MutableDataRaw(), values, sizeof(values));
  std::memset(dst.MutableDataRaw(), 0, sizeof(values));

  ASSERT_TRUE(data_transfer->CopyTensor(src, dst).IsOK()) << "CopyTensor failed for a CPU<->CPU copy.";
  EXPECT_EQ(std::memcmp(src.DataRaw(), dst.DataRaw(), sizeof(values)), 0)
      << "A CPU round-trip CopyTensor must preserve data exactly.";
}

// Invariant: read-only metadata queries are callable and self-consistent on a
// freshly constructed EP (no session or logger required), and the device-id
// accessor agrees with GetDevice().
TEST_P(EpConformanceTest, MetadataQueriesAreCallable) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  EXPECT_EQ(ep->GetDeviceId(), ep->GetDevice().Id())
      << "GetDeviceId() must agree with GetDevice().Id().";

  // These must simply be callable without crashing on a bare EP instance.
  (void)ep->ConcurrentRunSupported();
  (void)ep->GetTuningContext();
  (void)ep->GetOrtDeviceByMemType(OrtMemTypeDefault);
  (void)ep->IsGraphCaptureEnabled();
  (void)ep->ShouldConvertDataLayoutForOp(/*domain*/ "", /*op_type*/ "Conv", ep->GetPreferredLayout());
}

// Invariant: GetGraphCaptureNodeAssignmentPolicy() returns one of the defined
// OrtGraphCaptureNodeAssignmentPolicy values. The session dispatches on this
// while validating a graph for capture, so an out-of-range value is a bug.
// This is a pure query and is valid to call on every EP regardless of whether
// graph capture is enabled.
TEST_P(EpConformanceTest, GraphCaptureNodeAssignmentPolicyIsValid) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  const OrtGraphCaptureNodeAssignmentPolicy policy = ep->GetGraphCaptureNodeAssignmentPolicy();
  EXPECT_TRUE(policy == OrtGraphCaptureNodeAssignmentPolicy_ALL_NODES_ON_EP ||
              policy == OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES)
      << "GetGraphCaptureNodeAssignmentPolicy() returned an unknown policy value.";
}

// Invariant: a built-in EP returns no backing OrtEp. A PluginExecutionProvider
// returns the same non-null backing OrtEp across repeated queries.
TEST_P(EpConformanceTest, GetOrtEpMatchesProviderKind) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  const OrtEp* ort_ep = ep->GetOrtEp();
  if (GetParam().expects_plugin_ep) {
    ASSERT_NE(ort_ep, nullptr) << "A plugin EP must report its backing OrtEp.";
    EXPECT_EQ(ep->GetOrtEp(), ort_ep) << "A plugin EP must report a stable backing OrtEp.";
  } else {
    EXPECT_EQ(ort_ep, nullptr)
        << "A built-in EP must not report a backing OrtEp (that is reserved for plugin EPs).";
  }
}

// Invariant: GetEpContextNodes() reports no nodes on a freshly constructed EP.
// EPs populate this only when generating an EPContext cache model during
// compilation; with no compilation performed, the documented default is empty.
TEST_P(EpConformanceTest, EpContextNodesEmptyOnFreshEp) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  EXPECT_TRUE(ep->GetEpContextNodes().empty())
      << "A fresh EP (no compilation performed) must report no EPContext nodes.";
}

// Invariant: every preferred allocator reports self-consistent OrtMemoryInfo --
// a non-empty name and a valid allocator type. This metadata keys allocator
// lookup in the framework, so it must be well-formed for every EP. Only the
// backend-agnostic fields are checked; the raw memory is not touched here.
TEST_P(EpConformanceTest, PreferredAllocatorInfoIsConsistent) {
  auto ep = MakeEp();
  if (!ep) GTEST_SKIP() << GetParam().name << " EP not available in this environment.";

  for (const auto& alloc : ep->CreatePreferredAllocators()) {
    ASSERT_NE(alloc, nullptr) << "CreatePreferredAllocators() must not return null entries.";
    const OrtMemoryInfo& info = alloc->Info();
    EXPECT_FALSE(info.name.empty()) << "Allocator OrtMemoryInfo.name must not be empty.";
    EXPECT_NE(info.alloc_type, OrtInvalidAllocator)
        << "Allocator must report a valid OrtAllocatorType (not OrtInvalidAllocator).";
  }
}

INSTANTIATE_TEST_SUITE_P(
    EpContract, EpConformanceTest, testing::ValuesIn(GetEpConformanceParams()),
    [](const testing::TestParamInfo<EpConformanceParam>& info) { return info.param.name; });

}  // namespace test
}  // namespace onnxruntime
