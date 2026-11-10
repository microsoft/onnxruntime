// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_stateful.h"
#include "ep_test_hooks.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <new>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
struct StatefulContext {
  std::atomic<size_t> references{1};
  const OrtApi& api;
  const OrtLogger& logger;
  StatefulContext(const OrtApi& value, const OrtLogger& log) : api(value), logger(log) {}
  ~StatefulContext() {
    auto* status = api.Logger_LogMessage(&logger, ORT_LOGGING_LEVEL_INFO, "Releasing retained stateful context",
                                         ORT_FILE, __LINE__, __FUNCTION__);
    if (status) {
      std::fprintf(stderr, "Stateful context logger failed: %s\n", api.GetErrorMessage(status));
      api.ReleaseStatus(status);
    }
    RecordStatefulLifetime(1);
  }
};

void ORT_API_CALL Retain(void* value) noexcept {
  ++static_cast<StatefulContext*>(value)->references;
}

void ORT_API_CALL Release(void* value) noexcept {
  auto* context = static_cast<StatefulContext*>(value);
  if (--context->references == 0) delete context;
}

OrtStatus* ORT_API_CALL Cache(void*, const OrtStatefulCacheOperationRequest* request,
                              OrtStatefulCacheOperationResult* out) noexcept {
  RecordStatefulCall(request->operation);
  if (StatefulFailureOperation() == request->operation) {
    return request->api->CreateStatus(static_cast<OrtErrorCode>(StatefulFailureCode()), "test plugin cache failure");
  }
  return request->cpu_reference(request->cpu_context, out);
}

OrtStatus* ORT_API_CALL Greedy(void*, const OrtStatefulGreedyRequest* request, int64_t* out) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  RecordStatefulCall(ORT_STATEFUL_SAMPLE_GREEDY);
  if (StatefulFailureOperation() == ORT_STATEFUL_SAMPLE_GREEDY) {
    return request->api->CreateStatus(static_cast<OrtErrorCode>(StatefulFailureCode()), "test plugin greedy failure");
  }
  Ort::ConstValue logits(request->logits);
  const auto metadata = logits.GetTensorTypeAndShapeInfo();
  const auto shape = metadata.GetShape();
  const size_t width = static_cast<size_t>(shape.back());
  const size_t row = request->has_position ? static_cast<size_t>(request->position)
                     : shape.size() == 3   ? static_cast<size_t>(shape[1] - 1)
                                           : 0;
  const auto type = metadata.GetElementType();
  auto value = [&](size_t index) {
    return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? logits.GetTensorData<float>()[index]
                                                       : logits.GetTensorData<Ort::Float16_t>()[index].ToFloat();
  };
  const size_t count = metadata.GetElementCount();
  for (size_t i = 0; i < count; ++i) {
    if (std::isnan(value(i))) return request->api->CreateStatus(ORT_INVALID_ARGUMENT, "Plugin rejects NaN logits");
  }
  size_t best = 0;
  for (size_t i = 1; i < width; ++i) {
    if (value(row * width + i) > value(row * width + best)) best = i;
  }
  *out = static_cast<int64_t>(best);
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

// Frozen 1.23 prefix, allocated directly against an inaccessible guard page.
struct LegacyEp23 {
  uint32_t ort_version_supported;
  decltype(OrtEp::GetName) GetName;
  decltype(OrtEp::GetCapability) GetCapability;
  decltype(OrtEp::Compile) Compile;
  decltype(OrtEp::ReleaseNodeComputeInfos) ReleaseNodeComputeInfos;
  decltype(OrtEp::GetPreferredDataLayout) GetPreferredDataLayout;
  decltype(OrtEp::ShouldConvertDataLayoutForOp) ShouldConvertDataLayoutForOp;
  decltype(OrtEp::SetDynamicOptions) SetDynamicOptions;
  decltype(OrtEp::OnRunStart) OnRunStart;
  decltype(OrtEp::OnRunEnd) OnRunEnd;
  decltype(OrtEp::CreateAllocator) CreateAllocator;
  decltype(OrtEp::CreateSyncStreamForDevice) CreateSyncStreamForDevice;
  decltype(OrtEp::GetCompiledModelCompatibilityInfo) GetCompiledModelCompatibilityInfo;
};
static_assert(sizeof(LegacyEp23) == offsetof(OrtEp, GetKernelRegistry));

struct Allocation {
  void* base;
  size_t bytes;
};

const char* ORT_API_CALL LegacyName(const OrtEp*) noexcept { return "example_ep"; }
OrtStatus* ORT_API_CALL LegacyCapability(OrtEp*, const OrtGraph*, OrtEpGraphSupportInfo*) noexcept {
  return nullptr;
}
}  // namespace

OrtStatus* RegisterExampleStatefulProvider(OrtEp* ep, ApiPtrs apis, const OrtLogger& logger,
                                           const std::string& mode) {
  if (mode == "none") return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  if (mode != "full" && mode != "sampler" && mode != "append" && mode != "bad_version" &&
      mode != "short" && mode != "extended" && mode != "missing_callback" &&
      mode != "duplicate" && mode != "wrong_ep" && mode != "create_fail") {
    return apis.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Unknown example stateful mode");
  }
  std::unique_ptr<StatefulContext, decltype(&Release)> context(new StatefulContext(apis.ort_api, logger), &Release);
  struct ExtendedDescriptor {
    OrtStatefulProviderDescriptor descriptor;
    uint64_t tail;
  } extended{};
  auto& descriptor = extended.descriptor;
  const uint64_t floats = (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) |
                          (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  descriptor.struct_size = sizeof(descriptor);
  descriptor.capabilities = {sizeof(OrtStatefulProviderCapabilities), 1,
                             mode == "sampler" ? uint64_t{64} : mode == "append" ? uint64_t{8}
                                                                                 : uint64_t{127},
                             floats | (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16), floats, 1, 0};
  descriptor.context = context.get();
  descriptor.CacheOperation = mode == "sampler" ? nullptr : &Cache;
  descriptor.Greedy = mode == "append" ? nullptr : &Greedy;
  descriptor.Retain = &Retain;
  descriptor.Release = &Release;
  if (mode == "bad_version") descriptor.capabilities.version = 99;
  if (mode == "short") descriptor.struct_size = offsetof(OrtStatefulProviderDescriptor, Release);
  if (mode == "extended") descriptor.struct_size = sizeof(extended);
  if (mode == "missing_callback") descriptor.Greedy = nullptr;
  OrtEp other{};
  Ort::RegisterStatefulProvider(mode == "wrong_ep" ? &other : ep, descriptor);
  if (mode == "duplicate") Ort::RegisterStatefulProvider(ep, descriptor);
  if (mode == "create_fail") return apis.ort_api.CreateStatus(ORT_EP_FAIL, "Injected failure after registration");
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* CreateLegacyExampleEp23(ApiPtrs apis, const char* name, OrtEp** out) {
  if (std::strcmp(name, "example_ep") != 0) {
    return apis.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Legacy fixture uses the example_ep registration");
  }
#ifdef _WIN32
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  const size_t page = info.dwPageSize;
  void* base = VirtualAlloc(nullptr, page * 2, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (!base) return apis.ort_api.CreateStatus(ORT_EP_FAIL, "Legacy guard allocation failed");
  DWORD previous;
  if (!VirtualProtect(static_cast<char*>(base) + page, page, PAGE_NOACCESS, &previous)) {
    VirtualFree(base, 0, MEM_RELEASE);
    return apis.ort_api.CreateStatus(ORT_EP_FAIL, "Legacy guard protection failed");
  }
#else
  const long page_value = sysconf(_SC_PAGESIZE);
  if (page_value <= 0) return apis.ort_api.CreateStatus(ORT_EP_FAIL, "Legacy page size failed");
  const size_t page = static_cast<size_t>(page_value);
  void* base = mmap(nullptr, page * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (base == MAP_FAILED) return apis.ort_api.CreateStatus(ORT_EP_FAIL, "Legacy guard allocation failed");
  if (mprotect(static_cast<char*>(base) + page, page, PROT_NONE) != 0) {
    munmap(base, page * 2);
    return apis.ort_api.CreateStatus(ORT_EP_FAIL, "Legacy guard protection failed");
  }
#endif
  auto* position = static_cast<char*>(base) + page - sizeof(LegacyEp23);
  new (position - sizeof(Allocation)) Allocation{base, page * 2};
  auto* ep = new (position) LegacyEp23{};
  ep->ort_version_supported = 23;
  ep->GetName = &LegacyName;
  ep->GetCapability = &LegacyCapability;
  RecordLegacySize(sizeof(LegacyEp23));
  *out = reinterpret_cast<OrtEp*>(ep);
  return nullptr;
}

void ReleaseLegacyExampleEp23(OrtEp* ep) noexcept {
  auto* position = reinterpret_cast<char*>(ep);
  const auto allocation = *reinterpret_cast<Allocation*>(position - sizeof(Allocation));
  reinterpret_cast<LegacyEp23*>(ep)->~LegacyEp23();
#ifdef _WIN32
  VirtualFree(allocation.base, 0, MEM_RELEASE);
#else
  munmap(allocation.base, allocation.bytes);
#endif
}
