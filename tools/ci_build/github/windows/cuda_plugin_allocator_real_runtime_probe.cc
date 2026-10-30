// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Standalone real-CUDA probe: link actual plugin allocator/arena objects and force-include ep/adapters.h.
// Requires an explicitly supplied CUDA runtime DLL; only the first allocation size is forced to SIZE_MAX.
#include <Windows.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "core/providers/cuda/plugin/cuda_arena.h"

struct OrtLogger {};
struct OrtStatus {
  OrtErrorCode code;
  std::string message;
};

namespace {
#define REAL_CUDA_FUNCTIONS(X) \
  X(cudaGetDevice)             \
  X(cudaSetDevice)             \
  X(cudaFree)                  \
  X(cudaFreeHost)              \
  X(cudaGetErrorName)          \
  X(cudaPeekAtLastError)       \
  X(cudaGetLastError)
#define DECLARE_REAL(name) decltype(&name) real_##name = nullptr;
REAL_CUDA_FUNCTIONS(DECLARE_REAL)
#undef DECLARE_REAL
using MallocFunction = cudaError_t(CUDARTAPI*)(void**, size_t);
using HostAllocFunction = cudaError_t(CUDARTAPI*)(void**, size_t, unsigned int);
MallocFunction real_cudaMalloc = nullptr;
HostAllocFunction real_cudaHostAlloc = nullptr;

struct Attempt {
  size_t logical_bytes = 0;
  size_t runtime_bytes = 0;
  cudaError_t status = cudaSuccess;
};
std::array<Attempt, 2> attempts;
size_t malloc_calls = 0;
size_t free_calls = 0;
cudaError_t free_status = cudaSuccess;
bool pinned_mode = false;

template <typename T>
T LoadFunction(HMODULE module, const char* name) {
  const auto address = GetProcAddress(module, name);
  if (address == nullptr) throw std::runtime_error(std::string("Missing cudart export: ") + name);
  return reinterpret_cast<T>(address);
}

template <typename Allocate>
cudaError_t ForwardAllocation(void** pointer, size_t bytes, const char* kind, Allocate allocate) {
  if (malloc_calls >= attempts.size()) std::abort();
  auto& attempt = attempts[malloc_calls++];
  attempt.logical_bytes = bytes;
  attempt.runtime_bytes = malloc_calls == 1 ? std::numeric_limits<size_t>::max() : bytes;
  attempt.status = allocate(attempt.runtime_bytes);
  std::cout << "forwarded_" << kind << " ordinal=" << malloc_calls << " logical_bytes=" << bytes
            << " runtime_bytes=" << attempt.runtime_bytes << " status=" << static_cast<int>(attempt.status)
            << " result=" << *pointer << std::endl;
  return attempt.status;
}

OrtStatus* ORT_API_CALL MemoryInfoName(const OrtMemoryInfo* info, const char** out) noexcept {
  *out = info->name.c_str();
  return nullptr;
}

const OrtEpApi* ORT_API_CALL EpApi() {
  static const OrtEpApi api{};
  return &api;
}

OrtStatus* ORT_API_CALL Severity(const OrtLogger*, OrtLoggingLevel* out) noexcept {
  *out = ORT_LOGGING_LEVEL_WARNING;
  return nullptr;
}

OrtStatus* ORT_API_CALL Log(const OrtLogger*, OrtLoggingLevel, const char* message,
                            const ORTCHAR_T*, int, const char*) noexcept {
  std::cout << message << std::endl;
  return nullptr;
}

OrtStatus* ORT_API_CALL CreateStatus(OrtErrorCode code, const char* message) noexcept {
  return new OrtStatus{code, message};
}

void ORT_API_CALL ReleaseStatus(OrtStatus* status) noexcept {
  delete status;
}
}  // namespace

extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device) {
  return real_cudaGetDevice(device);
}

extern "C" cudaError_t CUDARTAPI cudaSetDevice(int device) {
  return real_cudaSetDevice(device);
}

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** pointer, size_t bytes) {
  if (pinned_mode) std::abort();
  return ForwardAllocation(pointer, bytes, "cuda_malloc",
                           [&](size_t runtime_bytes) { return real_cudaMalloc(pointer, runtime_bytes); });
}

extern "C" cudaError_t CUDARTAPI cudaFree(void* pointer) {
  ++free_calls;
  free_status = real_cudaFree(pointer);
  return free_status;
}

extern "C" cudaError_t CUDARTAPI cudaHostAlloc(void** pointer, size_t bytes, unsigned int flags) {
  if (!pinned_mode) std::abort();
  return ForwardAllocation(pointer, bytes, "cuda_host_alloc",
                           [&](size_t runtime_bytes) { return real_cudaHostAlloc(pointer, runtime_bytes, flags); });
}

extern "C" cudaError_t CUDARTAPI cudaFreeHost(void* pointer) {
  ++free_calls;
  free_status = real_cudaFreeHost(pointer);
  return free_status;
}

extern "C" const char* CUDARTAPI cudaGetErrorName(cudaError_t error) {
  return real_cudaGetErrorName(error);
}

extern "C" cudaError_t CUDARTAPI cudaGetLastError() {
  return real_cudaGetLastError();
}

int wmain(int argc, wchar_t** argv) {
  if ((argc != 2 && argc != 3) || std::wstring_view(argv[1]).find(L":\\") != 1 ||
      (argc == 3 && std::wstring_view(argv[2]) != L"--pinned" && std::wstring_view(argv[2]) != L"--device")) {
    std::cerr << "usage: probe <absolute ARM64 cudart DLL path> [--device|--pinned]\n";
    return 64;
  }
  pinned_mode = argc == 3 && std::wstring_view(argv[2]) == L"--pinned";
  if (!SetEnvironmentVariableA("ORT_CUDA_GEMM_AUTO_TUNE_DIAGNOSTICS", "1")) return 65;
  HMODULE module = LoadLibraryExW(argv[1], nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (module == nullptr) {
    std::cerr << "LoadLibraryExW failed: " << GetLastError() << '\n';
    return 66;
  }
  // Keep the CUDA runtime loaded until process exit, including its TLS cleanup.
  try {
#define LOAD_REAL(name) real_##name = LoadFunction<decltype(real_##name)>(module, #name);
    REAL_CUDA_FUNCTIONS(LOAD_REAL)
    LOAD_REAL(cudaMalloc)
    LOAD_REAL(cudaHostAlloc)
#undef LOAD_REAL
    std::array<wchar_t, 32768> loaded_path{};
    if (GetModuleFileNameW(module, loaded_path.data(), static_cast<DWORD>(loaded_path.size())) == 0) return 67;
    std::wcout << L"REAL CUDA runtime loaded: " << loaded_path.data() << std::endl;
    OrtApi api{};
    api.MemoryInfoGetName = MemoryInfoName;
    api.GetEpApi = EpApi;
    api.Logger_GetLoggingSeverityLevel = Severity;
    api.Logger_LogMessage = Log;
    api.CreateStatus = CreateStatus;
    api.ReleaseStatus = ReleaseStatus;
    Ort::InitApi(&api);
    OrtLogger logger{};
    onnxruntime::ep::adapter::LoggingManager::CreateDefaultLogger(&logger);
    OrtMemoryInfo info{};
    info.name = pinned_mode ? "CudaPinned" : "Cuda";
    using namespace onnxruntime::cuda_plugin;
    cudaError_t after_retry = cudaSuccess;
    {
      AllocatorUniquePtr raw =
          pinned_mode
              ? AllocatorUniquePtr(new CudaPinnedAllocator(&info),
                                   [](OrtAllocator* allocator) { delete static_cast<CudaPinnedAllocator*>(allocator); })
              : AllocatorUniquePtr(new CudaDeviceAllocator(&info, 0),
                                   [](OrtAllocator* allocator) { delete static_cast<CudaDeviceAllocator*>(allocator); });
      ArenaConfig config;
      config.initial_chunk_size_bytes = 1024 * 1024;
      config.max_mem = 4 * 1024 * 1024;
      ArenaImpl arena(std::move(raw), config, api, logger);
      const auto before = real_cudaPeekAtLastError();
      std::cout << "before_allocation_peek=" << static_cast<int>(before) << std::endl;
      if (before != cudaSuccess) return 68;
      void* pointer = arena.Alloc(1024);
      after_retry = real_cudaPeekAtLastError();
      std::cout << "after_retry_peek=" << static_cast<int>(after_retry) << std::endl;
      if (pointer == nullptr || arena.RequestedSize(pointer) != 1024 || malloc_calls != 2) return 69;
      if (attempts[0].logical_bytes != 1048576 ||
          attempts[0].status != cudaErrorMemoryAllocation ||
          attempts[1].logical_bytes != 943872 || attempts[1].status != cudaSuccess) return 70;
      // Observation-only consumption in the probe, after the production allocation path has returned.
      const auto consumed = real_cudaGetLastError();
      const auto after_consume = real_cudaPeekAtLastError();
      std::cout << "probe_consumed=" << static_cast<int>(consumed)
                << " after_consume_peek=" << static_cast<int>(after_consume) << std::endl;
      if (consumed != after_retry || after_consume != cudaSuccess) return 71;
      arena.Free(pointer);
    }
    onnxruntime::ep::adapter::LoggingManager::DestroyDefaultLogger();
    if (free_calls != 1 || free_status != cudaSuccess) return 72;
    std::cout << "REAL runtime, actual production allocator/arena: pinned=" << pinned_mode
              << "; fallback succeeded; last_error="
              << static_cast<int>(after_retry) << "; historical GEMM attribution remains unproven.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Probe failed: " << error.what() << '\n';
    return 73;
  }
}
