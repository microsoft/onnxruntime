// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cufft_loader.h"

#ifndef USE_CUDA_MINIMAL

#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

// cuFFT's SONAME/DLL version tracks the CUDA major version it ships with:
// CUDA 12.x -> cuFFT 11, CUDA 13.x -> cuFFT 12. Select the name that matches the
// CUDA toolkit this library was built against (CUDA_VERSION comes from cuda_pch.h).
#if CUDA_VERSION >= 13000 && CUDA_VERSION < 14000
#ifdef _WIN32
constexpr const char* kCufftLibraryName = "cufft64_12.dll";
#else
constexpr const char* kCufftLibraryName = "libcufft.so.12";
#endif
#elif CUDA_VERSION >= 12000 && CUDA_VERSION < 13000
#ifdef _WIN32
constexpr const char* kCufftLibraryName = "cufft64_11.dll";
#else
constexpr const char* kCufftLibraryName = "libcufft.so.11";
#endif
#else
#error Unsupported CUDA_VERSION for dynamic cuFFT loading
#endif

#ifdef _WIN32
// Search the directories listed in the PATH environment variable for the given
// library and, if found, load it by its full path. Loading by full path (rather
// than letting the loader search) preserves the historical PATH-based cuFFT
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
  // Use LOAD_LIBRARY_SEARCH_DEFAULT_DIRS so cuFFT is resolved only from the
  // application directory, %WINDIR%\System32, and directories added via
  // AddDllDirectory/SetDefaultDllDirectories. This deliberately excludes the
  // current working directory from the search order to avoid loading an
  // attacker-controlled DLL from the process CWD.
  HMODULE handle = LoadLibraryExA(candidate.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (handle == nullptr) {
    // Fall back to searching the directories listed in PATH (loading by full
    // path), matching the pre-existing OS-loader behavior when cuFFT was a
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

namespace onnxruntime::cuda {

CufftLibrary& CufftLibrary::Get() {
  static CufftLibrary library;
  return library;
}

bool CufftLibrary::Available() {
  return EnsureLoaded();
}

std::string CufftLibrary::Error() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return error_.empty() ? CufftUnavailableErrorString() : error_;
}

bool CufftLibrary::EnsureLoaded() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (load_attempted_) {
    return available_;
  }

  load_attempted_ = true;
  handle_ = LoadLibraryCandidate(kCufftLibraryName, error_);
  if (handle_ != nullptr) {
    available_ = true;
    error_.clear();
    return true;
  }

  available_ = false;
  if (error_.empty()) {
    error_ = "cuFFT library was not found";
  }
  return false;
}

void* CufftLibrary::ResolveSymbol(const char* symbol) {
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

const char* CufftUnavailableErrorString() {
  return "cuFFT is not available. Install cuFFT or update the system library search path to enable FFT (Rfft/Irfft) operators.";
}

}  // namespace onnxruntime::cuda

#endif  // USE_CUDA_MINIMAL
