// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <new>
#include <optional>
#include <system_error>

namespace {
struct Barrier {
  std::mutex mutex;
  std::condition_variable changed;
  int entered = 0;
  int blocked = 0;
  bool released = false;
} barrier;

struct Kernel {
  explicit Kernel(const OrtApi& api) : api(api) {}
  const OrtApi& api;

  OrtStatus* ComputeV2(OrtKernelContext* context) {
    try {
      std::unique_lock lock(barrier.mutex);
      ++barrier.entered;
      ++barrier.blocked;
      barrier.changed.notify_all();
      const bool ready = barrier.changed.wait_for(lock, std::chrono::seconds(10), [] { return barrier.released; });
      --barrier.blocked;
      if (!ready) return api.CreateStatus(ORT_FAIL, "Stateful test barrier timed out");
    } catch (const std::system_error& error) {
      return api.CreateStatus(ORT_FAIL, error.what());
    }
    const OrtValue* input = nullptr;
    OrtValue* output = nullptr;
    const void* source = nullptr;
    void* destination = nullptr;
    if (auto* status = api.KernelContext_GetInput(context, 0, &input)) return status;
    if (auto* status = api.GetTensorData(input, &source)) return status;
    if (auto* status = api.KernelContext_GetOutput(context, 0, nullptr, 0, &output)) return status;
    if (auto* status = api.GetTensorMutableData(output, &destination)) return status;
    *static_cast<float*>(destination) = *static_cast<const float*>(source);
    return nullptr;
  }
};

struct Operation : Ort::CustomOpBase<Operation, Kernel, true> {
  const char* GetName() const { return "StatefulBarrier"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 1; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo*, void** out) const {
    auto* kernel = new (std::nothrow) Kernel(api);
    if (!kernel) return api.CreateStatus(ORT_FAIL, "Stateful test kernel allocation failed");
    *out = kernel;
    return nullptr;
  }
} operation;
}  // namespace

extern "C" ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  try {
    static std::mutex mutex;
    static std::optional<Ort::CustomOpDomain> domain;
    std::lock_guard lock(mutex);
    if (!domain) {
      Ort::CustomOpDomain created("test.stateful");
      created.Add(&operation);
      domain.emplace(std::move(created));
    }
    Ort::UnownedSessionOptions(options).Add(*domain);
    return nullptr;
  } catch (const Ort::Exception& error) {
    return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(), error.what());
  } catch (const std::system_error& error) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
  } catch (const std::bad_alloc&) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, "Stateful test domain allocation failed");
  }
}

extern "C" ORT_EXPORT int StatefulBarrierReset() noexcept {
  try {
    std::lock_guard lock(barrier.mutex);
    if (barrier.blocked != 0) return -1;
    barrier.entered = 0;
    barrier.released = false;
    return 0;
  } catch (const std::system_error& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return -1;
  }
}

extern "C" ORT_EXPORT int StatefulBarrierWait(int count, int milliseconds) noexcept {
  try {
    std::unique_lock lock(barrier.mutex);
    return barrier.changed.wait_for(lock, std::chrono::milliseconds(milliseconds),
                                    [count] { return barrier.entered >= count; })
               ? 1
               : 0;
  } catch (const std::system_error& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return -1;
  }
}

extern "C" ORT_EXPORT int StatefulBarrierBlocked() noexcept {
  try {
    std::lock_guard lock(barrier.mutex);
    return barrier.blocked;
  } catch (const std::system_error& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return -1;
  }
}

extern "C" ORT_EXPORT int StatefulBarrierRelease() noexcept {
  try {
    std::lock_guard lock(barrier.mutex);
    barrier.released = true;
    barrier.changed.notify_all();
    return 0;
  } catch (const std::system_error& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return -1;
  }
}
