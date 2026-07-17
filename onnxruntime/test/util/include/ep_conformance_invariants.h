// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Shared Execution Provider conformance invariants.
//
// These free functions encode the backend-agnostic invariants that EVERY
// execution provider is expected to satisfy. They are the single source of truth
// shared by two suites so the internal-interface and plugin-interface checks
// cannot drift apart:
//   - The built-in EP suite (test/framework/execution_provider_conformance_test.cc)
//     exercises the in-tree IExecutionProvider implementations.
//   - The plugin EP suite (test/providers/ep_conformance_plugin_test.cc) exercises a
//     dynamically-loaded plugin EP -- i.e. an OrtEp reached through the
//     PluginExecutionProvider wrapper -- which is the recommended path for external
//     EP authors.
//
// Only documented, backend-agnostic contracts are asserted here. Memory that is not
// CPU-accessible is never dereferenced; such checks are guarded by
// OrtDevice::UsesCpuMemory().
//
// Each function issues gtest expectations/assertions directly. Because some of them
// call GTEST_SKIP() when a backend-agnostic precondition is not met (e.g. the EP
// exposes no CPU-accessible allocator), a function must be invoked as the LAST
// statement of a test body so the skip cleanly ends the test.

#include <cstring>
#include <functional>
#include <memory>
#include <string_view>

#include "gtest/gtest.h"

#include "core/framework/allocator.h"
#include "core/framework/data_transfer.h"
#include "core/framework/data_types.h"
#include "core/framework/execution_provider.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace test {
namespace ep_conformance {

// Factory that constructs a fresh EP instance under test. Used by invariants that
// need to compare independent instances produced by the same source.
using MakeEpFn = std::function<std::unique_ptr<IExecutionProvider>()>;

// Invariant: Type() is non-empty and stable -- both across repeated calls on a
// single instance and across independent instances from the same source. The
// framework keys kernel registries and node assignment on this string.
inline void CheckTypeIsNonEmptyAndStable(IExecutionProvider& ep, const MakeEpFn& make_ep, std::string_view label) {
  const std::string type = ep.Type();
  EXPECT_FALSE(type.empty()) << label << ": IExecutionProvider::Type() must not be empty.";
  EXPECT_EQ(type, ep.Type()) << label << ": Type() must be stable across calls on the same instance.";

  auto ep2 = make_ep();
  ASSERT_NE(ep2, nullptr) << label << ": the EP source must produce a second instance.";
  EXPECT_EQ(type, ep2->Type()) << label << ": Type() must be identical for instances from the same source.";
}

// Invariant: GetPreferredLayout() returns one of the defined DataLayout values.
// Layout transformation dispatches on this, so an out-of-range value is a bug.
inline void CheckPreferredLayoutIsValid(IExecutionProvider& ep, std::string_view label) {
  const DataLayout layout = ep.GetPreferredLayout();
  EXPECT_TRUE(layout == DataLayout::NCHW || layout == DataLayout::NHWC)
      << label << ": GetPreferredLayout() returned an unknown DataLayout value.";
}

// Invariant: the CPU mem types always map to CPU-accessible memory. The framework's
// input/output staging copies depend on this for every EP.
inline void CheckCpuMemTypesMapToCpuAccessibleDevice(IExecutionProvider& ep, std::string_view label) {
  EXPECT_TRUE(ep.GetOrtDeviceByMemType(OrtMemTypeCPUInput).UsesCpuMemory())
      << label << ": OrtMemTypeCPUInput must map to CPU-accessible memory.";
  EXPECT_TRUE(ep.GetOrtDeviceByMemType(OrtMemTypeCPUOutput).UsesCpuMemory())
      << label << ": OrtMemTypeCPUOutput must map to CPU-accessible memory.";
}

// Invariant: CreatePreferredAllocators() never yields a null allocator and is
// repeatable. The header documents it as a stateless factory, so a second call must
// produce an equivalently-sized set.
inline void CheckPreferredAllocatorsAreNonNullAndRepeatable(IExecutionProvider& ep, std::string_view label) {
  auto allocators = ep.CreatePreferredAllocators();
  for (const auto& alloc : allocators) {
    EXPECT_NE(alloc, nullptr) << label << ": CreatePreferredAllocators() must not return null entries.";
  }

  auto allocators2 = ep.CreatePreferredAllocators();
  EXPECT_EQ(allocators.size(), allocators2.size())
      << label << ": CreatePreferredAllocators() must be repeatable (documented as stateless).";
}

// Invariant: each CPU-accessible preferred allocator hands back usable memory: a
// non-zero allocation yields a non-null, host-writable and -readable pointer that can
// be freed. Device allocators are intentionally excluded here -- their raw Alloc/Free
// lifecycle is backend-specific -- and are covered by
// CheckPreferredAllocatorsAreNonNullAndRepeatable instead.
inline void CheckPreferredAllocatorsAllocateUsableMemory(IExecutionProvider& ep, std::string_view label) {
  auto allocators = ep.CreatePreferredAllocators();
  if (allocators.empty()) {
    GTEST_SKIP() << label << " EP exposes no preferred allocators.";
  }

  constexpr size_t kBytes = 256;
  size_t exercised = 0;
  for (const auto& alloc : allocators) {
    ASSERT_NE(alloc, nullptr);

    // Standalone Alloc()/Free() is only a backend-agnostic contract for
    // CPU-accessible allocators. A device allocator may hand out memory with a
    // backend-specific lifecycle that this test cannot drive: e.g. the WebGPU
    // GpuBufferAllocator creates buffers mapped at creation that must be unmapped
    // through the buffer manager before Free(), so Free()-ing a freshly allocated
    // buffer throws. Skip such allocators; they are covered by
    // CheckPreferredAllocatorsAreNonNullAndRepeatable.
    if (!alloc->Info().device.UsesCpuMemory()) {
      continue;
    }

    void* p = alloc->Alloc(kBytes);
    ASSERT_NE(p, nullptr) << label << ": Alloc(" << kBytes << ") returned null for allocator on "
                          << alloc->Info().device.ToString();

    std::memset(p, 0xAB, kBytes);
    const auto* bytes = static_cast<const unsigned char*>(p);
    EXPECT_EQ(bytes[0], 0xAB);
    EXPECT_EQ(bytes[kBytes - 1], 0xAB);
    alloc->Free(p);
    ++exercised;
  }

  if (exercised == 0) {
    GTEST_SKIP() << label << " EP exposes no CPU-accessible preferred allocator to exercise.";
  }
}

// Invariant: GetDataTransfer() is optional (may be null). When provided, and it
// advertises the ability to copy within a CPU-accessible device, a CopyTensor
// round-trip must preserve the data exactly.
inline void CheckDataTransferCpuRoundTripPreservesData(IExecutionProvider& ep, std::string_view label) {
  auto data_transfer = ep.GetDataTransfer();
  if (!data_transfer) {
    GTEST_SKIP() << label << " EP provides no IDataTransfer (allowed by the contract).";
  }

  // Host the tensors on a CPU-accessible preferred allocator.
  AllocatorPtr cpu_alloc;
  for (auto& alloc : ep.CreatePreferredAllocators()) {
    if (alloc && alloc->Info().device.UsesCpuMemory()) {
      cpu_alloc = alloc;
      break;
    }
  }
  if (!cpu_alloc) {
    GTEST_SKIP() << label << " EP has no CPU-accessible allocator to drive the round-trip.";
  }

  const OrtDevice& cpu_device = cpu_alloc->Info().device;
  if (!data_transfer->CanCopy(cpu_device, cpu_device)) {
    GTEST_SKIP() << label << " EP data transfer does not advertise CPU<->CPU copy.";
  }

  const TensorShape shape({2, 3});
  Tensor src(DataTypeImpl::GetType<float>(), shape, cpu_alloc);
  Tensor dst(DataTypeImpl::GetType<float>(), shape, cpu_alloc);

  const float values[] = {1.f, -2.f, 3.5f, 4.f, 5.f, -6.25f};
  constexpr size_t kNumValues = sizeof(values) / sizeof(values[0]);
  ASSERT_EQ(static_cast<size_t>(shape.Size()), kNumValues);
  std::memcpy(src.MutableDataRaw(), values, sizeof(values));
  std::memset(dst.MutableDataRaw(), 0, sizeof(values));

  ASSERT_TRUE(data_transfer->CopyTensor(src, dst).IsOK()) << label << ": CopyTensor failed for a CPU<->CPU copy.";
  EXPECT_EQ(std::memcmp(src.DataRaw(), dst.DataRaw(), sizeof(values)), 0)
      << label << ": a CPU round-trip CopyTensor must preserve data exactly.";
}

// Invariant: read-only metadata queries are callable and self-consistent on a
// freshly constructed EP (no session or logger required), and the device-id accessor
// agrees with GetDevice().
inline void CheckMetadataQueriesAreCallable(IExecutionProvider& ep, std::string_view label) {
  EXPECT_EQ(ep.GetDeviceId(), ep.GetDevice().Id()) << label << ": GetDeviceId() must agree with GetDevice().Id().";

  // These must simply be callable without crashing on a bare EP instance.
  (void)ep.ConcurrentRunSupported();
  (void)ep.GetTuningContext();
  (void)ep.GetOrtDeviceByMemType(OrtMemTypeDefault);
  (void)ep.IsGraphCaptureEnabled();
  (void)ep.ShouldConvertDataLayoutForOp(/*domain*/ "", /*op_type*/ "Conv", ep.GetPreferredLayout());
}

// Invariant: GetGraphCaptureNodeAssignmentPolicy() returns one of the defined
// OrtGraphCaptureNodeAssignmentPolicy values. The session dispatches on this while
// validating a graph for capture, so an out-of-range value is a bug. This is a pure
// query and is valid to call on every EP regardless of whether graph capture is
// enabled.
inline void CheckGraphCaptureNodeAssignmentPolicyIsValid(IExecutionProvider& ep, std::string_view label) {
  const OrtGraphCaptureNodeAssignmentPolicy policy = ep.GetGraphCaptureNodeAssignmentPolicy();
  EXPECT_TRUE(policy == OrtGraphCaptureNodeAssignmentPolicy_ALL_NODES_ON_EP ||
              policy == OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES)
      << label << ": GetGraphCaptureNodeAssignmentPolicy() returned an unknown policy value.";
}

// Invariant: GetOrtEp() reflects the provider kind. A built-in EP returns no backing
// OrtEp; a PluginExecutionProvider returns the same non-null backing OrtEp across
// repeated queries.
inline void CheckGetOrtEpMatchesProviderKind(IExecutionProvider& ep, bool expects_plugin_ep, std::string_view label) {
  const OrtEp* ort_ep = ep.GetOrtEp();
  if (expects_plugin_ep) {
    ASSERT_NE(ort_ep, nullptr) << label << ": a plugin EP must report its backing OrtEp.";
    EXPECT_EQ(ep.GetOrtEp(), ort_ep) << label << ": a plugin EP must report a stable backing OrtEp.";
  } else {
    EXPECT_EQ(ort_ep, nullptr)
        << label << ": a built-in EP must not report a backing OrtEp (that is reserved for plugin EPs).";
  }
}

// Invariant: GetEpContextNodes() reports no nodes on a freshly constructed EP. EPs
// populate this only when generating an EPContext cache model during compilation;
// with no compilation performed, the documented default is empty.
inline void CheckEpContextNodesEmptyOnFreshEp(IExecutionProvider& ep, std::string_view label) {
  EXPECT_TRUE(ep.GetEpContextNodes().empty())
      << label << ": a fresh EP (no compilation performed) must report no EPContext nodes.";
}

// Invariant: every preferred allocator reports self-consistent OrtMemoryInfo -- a
// non-empty name and a valid allocator type. This metadata keys allocator lookup in
// the framework, so it must be well-formed for every EP. Only the backend-agnostic
// fields are checked; the raw memory is not touched here.
inline void CheckPreferredAllocatorInfoIsConsistent(IExecutionProvider& ep, std::string_view label) {
  for (const auto& alloc : ep.CreatePreferredAllocators()) {
    ASSERT_NE(alloc, nullptr) << label << ": CreatePreferredAllocators() must not return null entries.";
    const OrtMemoryInfo& info = alloc->Info();
    EXPECT_FALSE(info.name.empty()) << label << ": allocator OrtMemoryInfo.name must not be empty.";
    EXPECT_NE(info.alloc_type, OrtInvalidAllocator)
        << label << ": allocator must report a valid OrtAllocatorType (not OrtInvalidAllocator).";
  }
}

}  // namespace ep_conformance
}  // namespace test
}  // namespace onnxruntime
