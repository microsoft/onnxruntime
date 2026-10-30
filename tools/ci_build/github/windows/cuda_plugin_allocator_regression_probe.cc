// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Standalone CPU-only probe: link actual plugin allocator/arena objects and force-include ep/adapters.h.
// CUDA entrypoints below inject failures; no CUDA runtime or provider DLL is linked or loaded.
#include <Windows.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "core/providers/cuda/plugin/cuda_arena.h"

struct OrtLogger {};
struct OrtStatus {
  OrtErrorCode code;
  std::string message;
};

namespace {
thread_local cudaError_t last_error = cudaSuccess;
cudaError_t injected_error = cudaErrorMemoryAllocation;
size_t malloc_calls = 0;
size_t free_calls = 0;
size_t error_consumptions = 0;
size_t raw_record_count = 0;
std::array<size_t, 2> attempted_bytes{};
std::array<std::string, 2> raw_records;

OrtStatus* ORT_API_CALL MemoryInfoName(const OrtMemoryInfo*, const char** out) noexcept {
  *out = "Cuda";
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
  if (std::strstr(message, "[CUDA GEMM diagnostic] cuda_malloc") != nullptr) {
    if (raw_record_count >= raw_records.size()) std::abort();
    raw_records[raw_record_count++] = message;
  }
  std::cout << message << '\n';
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
  *device = 0;
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaSetDevice(int) {
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** pointer, size_t bytes) {
  if (malloc_calls >= attempted_bytes.size()) std::abort();
  attempted_bytes[malloc_calls++] = bytes;
  if (malloc_calls == 1) {
    *pointer = nullptr;
    last_error = injected_error;
    return injected_error;
  }
  *pointer = std::malloc(bytes);
  if (*pointer == nullptr) {
    last_error = cudaErrorMemoryAllocation;
    return cudaErrorMemoryAllocation;
  }
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaFree(void* pointer) {
  ++free_calls;
  std::free(pointer);
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaHostAlloc(void** pointer, size_t bytes, unsigned int) {
  *pointer = std::malloc(bytes);
  return *pointer == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaFreeHost(void* pointer) {
  std::free(pointer);
  return cudaSuccess;
}

extern "C" const char* CUDARTAPI cudaGetErrorName(cudaError_t error) {
  if (error == cudaSuccess) return "cudaSuccess";
  return error == cudaErrorMemoryAllocation ? "cudaErrorMemoryAllocation" : "cudaErrorInvalidValue";
}

extern "C" cudaError_t CUDARTAPI cudaGetLastError() {
  ++error_consumptions;
  const auto error = last_error;
  last_error = cudaSuccess;
  return error;
}

int main(int argc, char** argv) {
  if (argc != 3 ||
      (std::strcmp(argv[1], "--oom") != 0 && std::strcmp(argv[1], "--non-oom") != 0) ||
      (std::strcmp(argv[2], "--off") != 0 && std::strcmp(argv[2], "--on") != 0)) return 64;
  const bool oom = std::strcmp(argv[1], "--oom") == 0;
  const bool diagnostics = std::strcmp(argv[2], "--on") == 0;
  injected_error = oom ? cudaErrorMemoryAllocation : cudaErrorInvalidValue;
  if (!SetEnvironmentVariableA("ORT_CUDA_GEMM_AUTO_TUNE_DIAGNOSTICS", diagnostics ? "1" : "0")) return 65;
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
  using namespace onnxruntime::cuda_plugin;
  {
    AllocatorUniquePtr raw(
        new CudaDeviceAllocator(&info, 0),
        [](OrtAllocator* allocator) { delete static_cast<CudaDeviceAllocator*>(allocator); });
    ArenaConfig config;
    config.initial_chunk_size_bytes = 1024 * 1024;
    config.max_mem = 4 * 1024 * 1024;
    ArenaImpl arena(std::move(raw), config, api, logger);
    void* pointer = arena.Alloc(1024);
    std::cout << "CPU-only actual-object regression: injected=" << static_cast<int>(injected_error)
              << " diagnostics=" << diagnostics << " malloc_calls=" << malloc_calls
              << " error_consumptions=" << error_consumptions << " after_retry=" << static_cast<int>(last_error)
              << '\n';
    if (pointer == nullptr || arena.RequestedSize(pointer) != 1024 || malloc_calls != 2 ||
        attempted_bytes[0] != 1048576 || attempted_bytes[1] != 943872) return 66;
    if (last_error != (oom ? cudaSuccess : injected_error) || error_consumptions != (oom ? 1u : 0u)) return 67;
    if (raw_record_count != (diagnostics ? 2u : 0u)) return 68;
    if (diagnostics &&
        (raw_records[0].find(" cuda_status=" + std::to_string(static_cast<int>(injected_error))) == std::string::npos ||
         raw_records[1].find(" cuda_status=0") == std::string::npos)) return 69;
    arena.Free(pointer);
  }
  onnxruntime::ep::adapter::LoggingManager::DestroyDefaultLogger();
  return free_calls == 1 ? 0 : 70;
}
