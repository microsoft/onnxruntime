// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cudnn_loader.h"

#ifndef USE_CUDA_MINIMAL

#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

std::vector<std::string> GetCandidateLibraryNames() {
#ifdef _WIN32
  constexpr const char* kCudnnLibraryName = "cudnn64_9.dll";
#else
  constexpr const char* kCudnnLibraryName = "libcudnn.so.9";
  constexpr const char* kCudnnUnversionedLibraryName = "libcudnn.so";
#endif

  std::vector<std::string> candidates;
  candidates.push_back(kCudnnLibraryName);
#ifndef _WIN32
  candidates.push_back(kCudnnUnversionedLibraryName);
#endif
  return candidates;
}

void* LoadLibraryCandidate(const std::string& candidate, std::string& error) {
#ifdef _WIN32
  HMODULE handle = LoadLibraryA(candidate.c_str());
  if (handle == nullptr) {
    error = "LoadLibrary failed for " + candidate + " with error " + std::to_string(GetLastError());
  }
  return reinterpret_cast<void*>(handle);
#else
  dlerror();
  void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* dl_error = dlerror();
    error = "dlopen failed for " + candidate + ": " + (dl_error != nullptr ? dl_error : "unknown error");
  }
  return handle;
#endif
}

void* GetLibrarySymbol(void* handle, const char* symbol, std::string& error) {
#ifdef _WIN32
  void* address = reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol));
  if (address == nullptr) {
    error = "GetProcAddress failed for " + std::string(symbol) + " with error " + std::to_string(GetLastError());
  }
  return address;
#else
  dlerror();
  void* address = dlsym(handle, symbol);
  const char* dl_error = dlerror();
  if (address == nullptr || dl_error != nullptr) {
    error = "dlsym failed for " + std::string(symbol) + ": " + (dl_error != nullptr ? dl_error : "unknown error");
  }
  return address;
#endif
}

}  // namespace

#if defined(NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)
namespace cudnn_frontend {
#ifdef _WIN32
HMODULE cudnn_dlhandle = nullptr;
#else
void* cudnn_dlhandle = nullptr;
#endif
}  // namespace cudnn_frontend
#endif

namespace onnxruntime::cuda {

CudnnLibrary& CudnnLibrary::Get() {
  static CudnnLibrary library;
  return library;
}

bool CudnnLibrary::Available() {
  return EnsureLoaded();
}

const char* CudnnLibrary::Error() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return error_.empty() ? CudnnUnavailableErrorString() : error_.c_str();
}

void* CudnnLibrary::Handle() {
  return EnsureLoaded() ? handle_ : nullptr;
}

bool CudnnLibrary::EnsureLoaded() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (load_attempted_) {
    return available_;
  }

  load_attempted_ = true;
  std::string last_error;
  for (const auto& candidate : GetCandidateLibraryNames()) {
    handle_ = LoadLibraryCandidate(candidate, last_error);
    if (handle_ != nullptr) {
      available_ = true;
      error_.clear();
#if defined(NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)
#ifdef _WIN32
      cudnn_frontend::cudnn_dlhandle = reinterpret_cast<HMODULE>(handle_);
#else
      cudnn_frontend::cudnn_dlhandle = handle_;
#endif
#endif
      return true;
    }
  }

  available_ = false;
  error_ = last_error.empty() ? "cuDNN library was not found" : last_error;
  return false;
}

void* CudnnLibrary::ResolveSymbol(const char* symbol) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = symbols_.find(symbol);
    if (it != symbols_.end()) {
      return it->second;
    }
  }

  if (!EnsureLoaded()) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  std::string symbol_error;
  void* address = GetLibrarySymbol(handle_, symbol, symbol_error);
  if (address == nullptr) {
    available_ = false;
    error_ = symbol_error;
    return nullptr;
  }

  symbols_.emplace(symbol, address);
  return address;
}

const char* CudnnUnavailableErrorString() {
  return "cuDNN is not available. Install cuDNN, update the system library search path, or set enable_cudnn=0 to force native CUDA paths where available.";
}

}  // namespace onnxruntime::cuda

#endif  // USE_CUDA_MINIMAL
