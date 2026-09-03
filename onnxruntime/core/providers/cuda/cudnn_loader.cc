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

#ifdef _WIN32
// Search the directories listed in the PATH environment variable for the given
// library and, if found, load it by its full path. Loading by full path (rather
// than letting the loader search) preserves the historical PATH-based cuDNN
// discovery without ever loading from the current working directory, which the
// LOAD_LIBRARY_SEARCH_DEFAULT_DIRS-only search excludes for security reasons.
HMODULE LoadLibraryFromPathEnv(const std::string& candidate) {
  DWORD length = GetEnvironmentVariableA("PATH", nullptr, 0);
  if (length == 0) {
    return nullptr;
  }

  std::string path_value(length, '\0');
  length = GetEnvironmentVariableA("PATH", path_value.data(), length);
  if (length == 0) {
    return nullptr;
  }
  path_value.resize(length);

  size_t start = 0;
  while (start <= path_value.size()) {
    size_t end = path_value.find(';', start);
    if (end == std::string::npos) {
      end = path_value.size();
    }

    std::string dir = path_value.substr(start, end - start);
    start = end + 1;
    if (dir.empty()) {
      continue;
    }

    if (dir.back() != '\\' && dir.back() != '/') {
      dir.push_back('\\');
    }
    std::string full_path = dir + candidate;

    if (GetFileAttributesA(full_path.c_str()) == INVALID_FILE_ATTRIBUTES) {
      continue;
    }

    HMODULE handle = LoadLibraryExA(full_path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (handle != nullptr) {
      return handle;
    }
  }

  return nullptr;
}
#endif

void* LoadLibraryCandidate(const std::string& candidate, std::string& error) {
#ifdef _WIN32
  // Use LOAD_LIBRARY_SEARCH_DEFAULT_DIRS so cuDNN is resolved only from the
  // application directory, %WINDIR%\System32, and directories added via
  // AddDllDirectory/SetDefaultDllDirectories. This deliberately excludes the
  // current working directory from the search order to avoid loading an
  // attacker-controlled DLL from the process CWD.
  HMODULE handle = LoadLibraryExA(candidate.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (handle == nullptr) {
    // Fall back to searching the directories listed in PATH (loading by full
    // path), matching the pre-existing OS-loader behavior when cuDNN was a
    // direct import dependency. The current working directory is never searched.
    handle = LoadLibraryFromPathEnv(candidate);
  }
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

std::string CudnnLibrary::Error() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return error_.empty() ? CudnnUnavailableErrorString() : error_;
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
