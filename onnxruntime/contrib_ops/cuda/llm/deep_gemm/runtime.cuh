/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "jit_utils.cuh"
#include "scheduler.cuh"

namespace deep_gemm::jit {

static bool kJitDebugging = []() {
  char const* env_var = getenv("TRTLLM_DG_JIT_DEBUG");
  return env_var && (std::string(env_var) == "1" || std::string(env_var) == "true");
}();

static bool kJitUseNvcc = []() {
  char const* env_var = getenv("TRTLLM_DG_JIT_USE_NVCC");
  return env_var && (std::string(env_var) == "1" || std::string(env_var) == "true");
}();

static bool kJitDumpCubin = []() {
  char const* env_var = getenv("TRTLLM_DG_JIT_DUMP_CUBIN");
  return env_var && (std::string(env_var) == "1" || std::string(env_var) == "true");
}();

static std::string kKernelName = kJitUseNvcc ? "nvcc_kernel.cubin" : "nvrtc_kernel.cubin";

/**
 * C++ implementation of the Runtime class from runtime.py
 * Loads and executes JIT-compiled kernels
 */
class Runtime {
 public:
  Runtime(std::string const& path, std::vector<char> const& cubin, deep_gemm::GemmType gemm_type)
      : path_(path), cubin_(cubin), gemm_type_(gemm_type), lib_(nullptr), kernel_(nullptr) {
    DG_HOST_ASSERT(!cubin.empty() || isPathValid(path_));
  }

  ~Runtime() {
    if (lib_ != nullptr) {
      CHECK_CUDA(cuLibraryUnload(lib_));
    }
  }

  static bool isPathValid(std::string const& path) {
    // Check if path exists and is a directory
    if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path)) {
      return false;
    }

    // Check if all necessary files exist
    return std::filesystem::exists(std::filesystem::path(path) / kKernelName);
  }

  CUkernel getKernel() {
    // Load shared object if not already loaded
    if (kernel_ == nullptr) {
      if (cubin_.empty()) {
        std::filesystem::path cubinPath = std::filesystem::path(path_);
        cubinPath /= kKernelName;
        std::ifstream cubinFile(cubinPath.string(), std::ios::binary);
        cubin_ = std::vector<char>(std::istreambuf_iterator<char>(cubinFile), {});
      }

      CHECK_CUDA(cuLibraryLoadData(&lib_, cubin_.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

      unsigned int numKernels = 0;
      CHECK_CUDA(cuLibraryGetKernelCount(&numKernels, lib_));

      std::vector<CUkernel> kernels(numKernels);
      CHECK_CUDA(cuLibraryEnumerateKernels(kernels.data(), numKernels, lib_));

      for (auto kernel : kernels) {
        char const* kernelName;
        CHECK_CUDA(cuKernelGetName(&kernelName, kernel));
        std::string kernelNameStr(kernelName);
        if (kernelNameStr.find("fp8_gemm_kernel") != std::string::npos) {
          kernel_ = kernel;
          break;
        }
      }

      if (!kernel_) {
        throw std::runtime_error("Failed to find fp8_gemm_kernel");
      }
    }

    return kernel_;
  }

 private:
  std::string path_;
  std::vector<char> cubin_;
  CUlibrary lib_;
  CUkernel kernel_;
  deep_gemm::GemmType gemm_type_;
};

/**
 * C++ implementation of the RuntimeCache class from runtime.py
 * Caches Runtime instances by path
 */
class RuntimeCache {
 public:
  static RuntimeCache& getInstance() {
    static RuntimeCache instance;
    return instance;
  }

  Runtime* operator[](std::string const& path) {
    // Check if already in cache
    auto it = cache_.find(path);
    if (it != cache_.end()) {
      return it->second.get();
    }

    // Check if already compiled
    if (Runtime::isPathValid(path)) {
      // Parse path to get gemm type
      std::string gemm_type_str = path.substr(path.find_last_of('_') + 1);
      deep_gemm::GemmType gemm_type;
      if (gemm_type_str == "Normal") {
        gemm_type = deep_gemm::GemmType::Normal;
      } else if (gemm_type_str == "GroupedWithOffset") {
        gemm_type = deep_gemm::GemmType::GroupedWithOffset;
      } else if (gemm_type_str == "StridedBatched") {
        gemm_type = deep_gemm::GemmType::StridedBatched;
      } else {
        throw std::runtime_error("Unsupported gemm type: " + gemm_type_str);
      }

      auto runtime = std::make_unique<Runtime>(path, std::vector<char>(), gemm_type);
      Runtime* result = runtime.get();
      cache_[path] = std::move(runtime);
      return result;
    }

    return nullptr;
  }

  void set(std::string const& path, std::unique_ptr<Runtime>&& runtime) {
    cache_[path] = std::move(runtime);
  }

 private:
  // Private constructor for singleton pattern
  RuntimeCache() = default;

  // Delete copy constructor and assignment operator
  RuntimeCache(RuntimeCache const&) = delete;
  RuntimeCache& operator=(RuntimeCache const&) = delete;

  std::unordered_map<std::string, std::unique_ptr<Runtime>> cache_;
};

// Global function to access the singleton
RuntimeCache& getGlobalRuntimeCache() {
  return RuntimeCache::getInstance();
}

}  // namespace deep_gemm::jit
