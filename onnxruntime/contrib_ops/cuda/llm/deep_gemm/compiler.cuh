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

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "jit_utils.cuh"
#include "nvrtc.h"
#include "runtime.cuh"
#include "scheduler.cuh"

#ifdef _WIN32
#include <windows.h>
#endif

namespace deep_gemm::jit {

// Generate a unique ID for temporary directories to avoid collisions
std::string generateUniqueId() {
  // Use current time and random number to generate a unique ID
  static std::mt19937 gen(std::random_device{}());
  static std::uniform_int_distribution<> distrib(0, 999999);

  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto value = now_ms.time_since_epoch().count();

  // Use the static random generator
  int random_value = distrib(gen);

  return std::to_string(value) + "_" + std::to_string(random_value);
}

std::filesystem::path getDefaultUserDir() {
  static std::filesystem::path userDir;
  if (userDir.empty()) {
    char const* cacheDir = getenv("TRTLLM_DG_CACHE_DIR");
    if (cacheDir) {
      userDir = cacheDir;
      std::filesystem::create_directories(userDir);
    } else {
#ifdef _WIN32
      char const* appData = getenv("APPDATA");
      if (appData) {
        userDir = std::filesystem::path(appData) / "tensorrt_llm";
      } else {
        userDir = std::filesystem::temp_directory_path() / "tensorrt_llm";
      }
#else
      char const* homeDir = getenv("HOME");
      if (homeDir) {
        userDir = std::filesystem::path(homeDir) / ".tensorrt_llm";
      } else {
        userDir = std::filesystem::temp_directory_path() / "tensorrt_llm";
      }
#endif
    }
  }
  return userDir;
}

inline std::filesystem::path getTmpDir() {
  return getDefaultUserDir() / "tmp";
}

inline std::filesystem::path getCacheDir() {
  return getDefaultUserDir() / "cache";
}

std::string getNvccCompiler() {
  static std::string compiler;
  if (compiler.empty()) {
    // Check environment variable
    char const* envCompiler = getenv("TRTLLM_DG_NVCC_COMPILER");
    if (envCompiler) {
      compiler = envCompiler;
    } else {
      // Check CUDA_HOME
      char const* cudaHome = getenv("CUDA_HOME");
      if (cudaHome) {
        std::filesystem::path cudaPath(cudaHome);
#ifdef _WIN32
        compiler = (cudaPath / "bin" / "nvcc.exe").string();
#else
        compiler = (cudaPath / "bin" / "nvcc").string();
#endif
      } else {
// Default to system nvcc
#ifdef _WIN32
        compiler = "nvcc.exe";
#else
        compiler = "nvcc";
#endif
      }
    }
  }
  return compiler;
}

std::vector<std::filesystem::path> getJitIncludeDirs() {
  static std::vector<std::filesystem::path> includeDirs;
  if (includeDirs.empty()) {
    // Command to execute
    char const* cmd = "pip show tensorrt_llm 2>/dev/null";

    // Buffer to store the output
    std::array<char, 128> buffer;
    std::string result;

// Open pipe to command
#ifdef _MSC_VER
    FILE* pipe = _popen(cmd, "r");
#else
    FILE* pipe = popen(cmd, "r");
#endif

    if (pipe) {
      // Read the output
      while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
      }

// Close the pipe
#ifdef _MSC_VER
      _pclose(pipe);
#else
      pclose(pipe);
#endif

      // Parse the location using regex
      // `pip show tensorrt_llm` will output something like:
      // Location: /usr/local/lib/python3.12/dist-packages
      // Editable project location: /code
      std::regex locationRegex("(Location|Editable project location): (.+)");

      // Find all matches
      auto match_begin = std::sregex_iterator(result.begin(), result.end(), locationRegex);
      auto match_end = std::sregex_iterator();

      // Get the number of matches
      auto match_count = std::distance(match_begin, match_end);

      if (match_count > 0) {
        // Get the last match
        auto last_match_iter = match_begin;
        std::advance(last_match_iter, match_count - 1);

        // Get the path from the second capture group
        std::string location = last_match_iter->str(2);
        location.erase(location.find_last_not_of(" \n\r\t") + 1);

        // Set the include directory based on the package location
        includeDirs.push_back(std::filesystem::path(location) / "tensorrt_llm" / "include");

        if (!kJitUseNvcc) {
          includeDirs.push_back(
              std::filesystem::path(location) / "tensorrt_llm" / "include" / "cuda" / "include");
        }
      }
    } else {
      ORT_LLM_LOG_WARNING("Failed to find TensorRT-LLM installation, DeepGEMM will be disabled.");
    }
  }
  return includeDirs;
}

std::string generateKernel(uint32_t const shape_n, uint32_t const shape_k, uint32_t const block_m,
                           uint32_t const block_n, uint32_t const block_k, uint32_t const num_groups, uint32_t const num_stages,
                           uint32_t const num_tma_multicast, deep_gemm::GemmType const gemm_type, bool swapAB = false) {
  constexpr uint32_t kNumTMAThreads = 128;
  constexpr uint32_t kNumMathThreadsPerGroup = 128;

  std::string input_type;
  if (!swapAB) {
    switch (gemm_type) {
      case deep_gemm::GemmType::Normal:
        input_type = "NormalSchedulerInput";
        break;
      case deep_gemm::GemmType::GroupedContiguous:
        input_type = "GroupedContiguousSchedulerInput";
        break;
      case deep_gemm::GemmType::GroupedMasked:
        input_type = "GroupedMaskedSchedulerInput";
        break;
      case deep_gemm::GemmType::GroupedWithOffset:
        input_type = "GroupedWithOffsetSchedulerInput";
        break;
      case deep_gemm::GemmType::StridedBatched:
        input_type = "StridedBatchedSchedulerInput";
        break;
      default:
        throw std::runtime_error("Unsupported gemm type");
    }
  } else {
    switch (gemm_type) {
      case deep_gemm::GemmType::Normal:
        input_type = "NormalSchedulerInputSwapAB";
        break;
      case deep_gemm::GemmType::GroupedWithOffset:
        input_type = "GroupedWithOffsetSchedulerInputSwapAB";
        break;
      default:
        throw std::runtime_error("Unsupported gemm type");
    }
  }

  // Modify kernel name based on swapAB to determine which kernel function to use
  std::string kernel_name = swapAB ? "fp8_gemm_kernel_swapAB" : "fp8_gemm_kernel";
  std::string scheduler_name = swapAB ? "SchedulerSelectorSwapAB" : "SchedulerSelector";

  // Create the kernel source code using raw string literal
  std::string code = R"(
#ifdef __CUDACC_RTC__
#ifndef NVRTC_JIT_COMPILATION
#define NVRTC_JIT_COMPILATION
#endif

#include <deep_gemm/nvrtc_std.cuh>

#else

#include <string>
#include <cuda.h>

#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <deep_gemm/nvrtc_cutlass.cuh>
#include <deep_gemm/fp8_gemm_impl.cuh>

using namespace deep_gemm;

using SchedulerType =
typename )" + scheduler_name +
                     R"(<GemmType::)" + gemm_type_to_string(gemm_type) + R"(, )" + std::to_string(shape_n) + R"(, )" + std::to_string(shape_k) + R"(, )" + std::to_string(block_m) + R"(, )" + std::to_string(block_n) + R"(, )" + std::to_string(block_k) + R"(, )" + std::to_string(num_groups) + R"(, )" + std::to_string(num_tma_multicast) + R"(>::type;

__global__ void dummy_kernel() {
  void *ptr = (void *)&)" +
                     kernel_name + R"(<)" + std::to_string(shape_n) + R"(, )" + std::to_string(shape_k) + R"(, )" + std::to_string(block_m) + R"(, )" + std::to_string(block_n) + R"(, )" + std::to_string(block_k) + R"(, )" + std::to_string(num_groups) + R"(, )" + std::to_string(num_stages) + R"(, )" + std::to_string(kNumTMAThreads) + R"(, )" + std::to_string(kNumMathThreadsPerGroup) + R"(, )" + std::to_string(num_tma_multicast) + R"(, SchedulerType, )" + input_type + R"(>;
}
)";

  return code;
}

/**
 * C++ implementation of the Compiler class
 * Compiles CUDA code into CUBINs
 */
class Compiler {
 public:
  // Get singleton instance
  static Compiler& getInstance() {
    static Compiler instance;
    return instance;
  }

  [[nodiscard]] bool isValid() const {
    return !includeDirs_.empty();
  }

  // Build function
  Runtime* build(uint32_t const shape_n, uint32_t const shape_k, uint32_t const block_m, uint32_t const block_n,
                 uint32_t const block_k, uint32_t const num_groups, uint32_t const num_stages, uint32_t const num_tma_multicast,
                 deep_gemm::GemmType const gemm_type, bool swapAB = false) {
    int sm_version = onnxruntime::llm::common::getSMVersion();
    if (sm_version != 90) {
      ORT_THROW(
          "DeepGEMM only supports Hopper (SM90) architectures, but current device compute "
          "capability is %d.",
          sm_version);
    }

    // Build signature - simplified, no MD5 calculation
    std::string name = std::string(swapAB ? "gemm_swapAB_" : "gemm_") + std::to_string(shape_n) + "_" + std::to_string(shape_k) + "_" + std::to_string(block_m) + "_" + std::to_string(block_n) + "_" + std::to_string(block_k) + "_" + std::to_string(num_groups) + "_" + std::to_string(num_stages) + std::to_string(num_groups) + "_" + std::to_string(num_stages) + "_" + std::to_string(num_tma_multicast) + "_" + gemm_type_to_string(gemm_type);
    std::filesystem::path path = getCacheDir() / name;

    // Check runtime cache or file system hit
    auto& runtimeCache = getGlobalRuntimeCache();
    Runtime* cachedRuntime = runtimeCache[path.string()];
    if (cachedRuntime != nullptr) {
      if (kJitDebugging) {
        ORT_LLM_LOG_INFO("Using cached JIT runtime ") << name << " during build";
      }
      return cachedRuntime;
    }

    // Compiler flags
    std::vector<std::string> flags = {"-std=c++17", "--gpu-architecture=sm_90a", "--ptxas-options=-allow-expensive-optimizations=true",
                                      "--ptxas-options=--register-usage-level=10", "--diag-suppress=161,174,177,940",
                                      "-D__FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__=1", "-D__FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__=1"};

    if (kJitUseNvcc) {
      flags.push_back("-O3");
      flags.push_back("-cubin");
      flags.push_back("--expt-relaxed-constexpr");
      flags.push_back("--expt-extended-lambda");

      std::vector<std::string> cxxFlags = {"-fPIC", "-O3", "-Wno-deprecated-declarations", "-Wno-abi"};
      std::string cxxFlagsStr = "--compiler-options=";
      for (size_t i = 0; i < cxxFlags.size(); ++i) {
        cxxFlagsStr += cxxFlags[i];
        if (i < cxxFlags.size() - 1) {
          cxxFlagsStr += ",";
        }
      }
      flags.push_back(cxxFlagsStr);
    } else {
      flags.push_back("-default-device");
    }

    std::filesystem::path tmpPath = getTmpDir() / (name + "_" + generateUniqueId());
    std::filesystem::path cubinPath = path / kKernelName;
    std::filesystem::path tmpCubinPath = tmpPath / kKernelName;

    // Create the target directory if it doesn't exist
    if (kJitUseNvcc || kJitDumpCubin) {
      std::filesystem::create_directories(tmpPath);
      std::filesystem::create_directories(path);
    }

    for (auto const& dir : includeDirs_) {
      flags.push_back("-I" + dir.string());
    }

    // Print options if debug enabled
    if (kJitDebugging) {
      std::ostringstream oss;
      for (auto const& flag : flags) {
        oss << " " << flag;
      }
      ORT_LLM_LOG_INFO("Compiling JIT runtime ") << name << " with options: " << oss.str();
    }

    std::string code = generateKernel(
        shape_n, shape_k, block_m, block_n, block_k, num_groups, num_stages, num_tma_multicast, gemm_type, swapAB);

    if (kJitDebugging) {
      ORT_LLM_LOG_INFO("Generated kernel code:\n") << code;
    }

    if (kJitUseNvcc) {
      std::filesystem::path tmpSrcPath = tmpPath / "kernel.cu";

      // Write files
      std::ofstream srcFile(tmpSrcPath);
      srcFile << code;
      srcFile.close();

      // Build command
      std::vector<std::string> command = {getNvccCompiler(), tmpSrcPath.string(), "-o", tmpCubinPath.string()};
      command.insert(command.end(), flags.begin(), flags.end());

      // Execute command
      std::string cmd;
      for (auto const& arg : command) {
        cmd += arg + " ";
      }

      // Buffer to store the output
      std::array<char, 128> buffer;
      std::string result;

      // Time the compilation
      auto start = std::chrono::high_resolution_clock::now();

      // Open pipe to command
#ifdef _MSC_VER
      FILE* pipe = _popen(cmd.c_str(), "r");
#else
      FILE* pipe = popen(cmd.c_str(), "r");
#endif

      if (pipe) {
        // Read the output
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
          result += buffer.data();
        }

// Close the pipe
#ifdef _MSC_VER
        _pclose(pipe);
#else
        pclose(pipe);
#endif

        // Output result if debug enabled
        if (kJitDebugging) {
          auto end = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
          ORT_LLM_LOG_INFO("NVCC compilation took ") << duration.count() << " ms";
          ORT_LLM_LOG_INFO("Compilation log:\n") << result;
        }
      }
    } else {
      nvrtcProgram prog;
      CHECK_NVRTC(nvrtcCreateProgram(&prog, code.c_str(), "kernel.cu", 0, nullptr, nullptr));

      std::vector<char const*> options;
      for (auto const& flag : flags) {
        options.push_back(flag.c_str());
      }

      // Time the compilation
      auto start = std::chrono::high_resolution_clock::now();
      nvrtcResult compileResult = nvrtcCompileProgram(prog, options.size(), options.data());

      if (kJitDebugging) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        ORT_LLM_LOG_INFO("NVRTC compilation took ") << duration.count() << " ms";

        size_t logSize;
        CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &logSize));
        std::vector<char> log(logSize);
        CHECK_NVRTC(nvrtcGetProgramLog(prog, log.data()));
        ORT_LLM_LOG_INFO("Compilation log:\n") << log.data();
      }

      // Check if compilation succeeded
      if (compileResult != NVRTC_SUCCESS) {
        ORT_LLM_LOG_ERROR("NVRTC compilation failed");
        CHECK_NVRTC(nvrtcDestroyProgram(&prog));
        throw std::runtime_error("NVRTC compilation failed");
      }

      // Save CUBIN to a file
      size_t cubinSize;
      CHECK_NVRTC(nvrtcGetCUBINSize(prog, &cubinSize));
      std::vector<char> cubin(cubinSize);
      CHECK_NVRTC(nvrtcGetCUBIN(prog, cubin.data()));

      // Cache the runtime in memory by default
      if (!kJitDumpCubin) {
        auto runtime = std::make_unique<Runtime>(path.string(), cubin, gemm_type);
        Runtime* result = runtime.get();
        runtimeCache.set(path.string(), std::move(runtime));
        if (kJitDebugging) {
          ORT_LLM_LOG_INFO("Successfully cached JIT runtime ") << name << " in memory";
        }
        return result;
      }

      std::ofstream cubinFile(tmpCubinPath.string(), std::ios::binary);
      cubinFile.write(cubin.data(), static_cast<std::streamsize>(cubinSize));
      cubinFile.close();
      CHECK_NVRTC(nvrtcDestroyProgram(&prog));
    }

    // Copy the source and compiled files to the cache directory
    try {
      // Rename (atomic operation) to final locations
      std::filesystem::rename(tmpCubinPath, cubinPath);
      if (kJitDebugging) {
        ORT_LLM_LOG_INFO("Successfully copied kernel files to cache directory:") << path.string();
      }
    } catch (std::exception const& e) {
      ORT_LLM_LOG_ERROR("Warning: Failed to copy kernel files to cache:") << e.what();
    }

    // Clean up temporary directory after successful compilation
    try {
      std::filesystem::remove_all(tmpPath);
    } catch (std::exception const& e) {
      ORT_LLM_LOG_ERROR("Warning: Failed to clean up temporary directory:") << e.what();
    }

    // Create runtime and cache it
    auto runtime = std::make_unique<Runtime>(path.string(), std::vector<char>(), gemm_type);
    Runtime* result = runtime.get();
    runtimeCache.set(path.string(), std::move(runtime));
    return result;
  }

 private:
  std::vector<std::filesystem::path> includeDirs_;

  // Private constructor for singleton pattern
  Compiler()
      : includeDirs_(getJitIncludeDirs()) {
    // Create necessary directories
    if (kJitUseNvcc || kJitDumpCubin) {
      std::filesystem::create_directories(getTmpDir());
      std::filesystem::create_directories(getCacheDir());
    }
  }

  // Delete copy constructor and assignment operator
  Compiler(Compiler const&) = delete;
  Compiler& operator=(Compiler const&) = delete;
};

// Global function to access the Compiler singleton
inline Compiler& getGlobalCompiler() {
  return Compiler::getInstance();
}

}  // namespace deep_gemm::jit
