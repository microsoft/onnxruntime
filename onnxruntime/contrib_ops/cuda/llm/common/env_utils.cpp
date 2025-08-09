/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "envUtils.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/common/logger.h"
// #include "tensorrt_llm/common/stringUtils.h"
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>

namespace onnxruntime::llm::common {

std::optional<int32_t> getIntEnv(char const* name) {
  char const* const env = std::getenv(name);
  if (env == nullptr) {
    return std::nullopt;
  }
  int32_t const val = std::stoi(env);
  return {val};
};

std::optional<size_t> getUInt64Env(char const* name) {
  char const* const env = std::getenv(name);
  if (env == nullptr) {
    return std::nullopt;
  }
  size_t const val = std::stoull(env);
  return {val};
};

std::optional<std::string> getStrEnv(char const* name) {
  char const* const env = std::getenv(name);
  if (env == nullptr) {
    return std::nullopt;
  }
  return std::string(env);
}

// Returns true if the env variable exists and is set to "1"
bool getBoolEnv(char const* name) {
  char const* env = std::getenv(name);
  return env && env[0] == '1' && env[1] == '\0';
}

static std::string trim(std::string const& str) {
  size_t start = str.find_first_not_of(" \t\n\r");
  size_t end = str.find_last_not_of(" \t\n\r");
  return (start == std::string::npos || end == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

// Parse memory size
static size_t parseMemorySize(std::string const& input) {
  std::string str = trim(input);

  size_t unitPos = 0;
  while (unitPos < str.size() && (std::isdigit(str[unitPos]) || str[unitPos] == '.')) {
    ++unitPos;
  }

  // Split the numeric part and the unit part
  std::string numberPart = str.substr(0, unitPos);
  std::string unitPart = str.substr(unitPos);

  double value = 0;
  try {
    value = std::stod(numberPart);
  } catch (std::invalid_argument const& e) {
    throw std::invalid_argument("Invalid number format in memory size: " + input);
  }

  toLower(unitPart);
  size_t multiplier = 1;
  if (unitPart == "b") {
    multiplier = 1;
  } else if (unitPart == "kb") {
    multiplier = 1024;
  } else if (unitPart == "mb") {
    multiplier = 1024 * 1024;
  } else if (unitPart == "gb") {
    multiplier = 1024 * 1024 * 1024;
  } else if (unitPart == "tb") {
    multiplier = static_cast<size_t>(pow(1024.0, 4));
  } else {
    throw std::invalid_argument("Unknown unit in memory size: " + unitPart);
  }

  return static_cast<size_t>(value * multiplier);
}

// XQA kernels (optimized kernels for generation phase).
bool forceXQAKernels() {
  static bool const forceXQA = (getIntEnv("TRTLLM_FORCE_XQA").value_or(0) != 0) || getEnvForceDeterministicAttention();
  return forceXQA;
}

std::optional<bool> getEnvEnableXQAJIT() {
  static std::optional<bool> val = [] {
    std::optional<bool> val = std::nullopt;
    auto const tmp = getIntEnv("TRTLLM_ENABLE_XQA_JIT");
    if (tmp.has_value()) {
      val = static_cast<bool>(tmp.value());
    }
    return val;
  }();
  return val;
}

std::optional<int> getEnvXqaBlocksPerSequence() {
  static auto const xqaBlocksPerSeq = []() {
    auto const val = getIntEnv("TRTLLM_XQA_BLOCKS_PER_SEQUENCE");
    return (val.has_value() && *val <= 0) ? std::nullopt : val;
  }();
  return xqaBlocksPerSeq;
}

// Tune the number of blocks per sequence for accuracy/performance purpose.
bool getEnvMmhaMultiblockDebug() {
  static std::once_flag flag;
  static bool forceMmhaMaxSeqLenTile = false;
  std::call_once(flag,
                 [&] {
                   char const* enable_mmha_debug_var = std::getenv("TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG");
                   if (enable_mmha_debug_var) {
                     if (enable_mmha_debug_var[0] == '1' && enable_mmha_debug_var[1] == '\0') {
                       forceMmhaMaxSeqLenTile = true;
                     }
                   }
                 });
  return forceMmhaMaxSeqLenTile;
}

int getEnvMmhaBlocksPerSequence() {
  static std::once_flag flag;
  static int mmhaBlocksPerSequence = 0;
  std::call_once(flag,
                 [&]() {
                   char const* mmhaBlocksPerSequenceEnv = std::getenv("TRTLLM_MMHA_BLOCKS_PER_SEQUENCE");
                   if (mmhaBlocksPerSequenceEnv) {
                     mmhaBlocksPerSequence = std::atoi(mmhaBlocksPerSequenceEnv);
                     if (mmhaBlocksPerSequence <= 0) {
                       ORT_LLM_LOG_WARNING(
                           "Invalid value for TRTLLM_MMHA_BLOCKS_PER_SEQUENCE. Will use default values instead!");
                     }
                   }
                 });

  return mmhaBlocksPerSequence;
}

int getEnvMmhaKernelBlockSize() {
  static std::once_flag flag;
  static int mmhaKernelBlockSize = 0;

  std::call_once(flag,
                 [&]() {
                   char const* mmhaKernelBlockSizeEnv = std::getenv("TRTLLM_MMHA_KERNEL_BLOCK_SIZE");
                   if (mmhaKernelBlockSizeEnv) {
                     mmhaKernelBlockSize = std::atoi(mmhaKernelBlockSizeEnv);
                     if (mmhaKernelBlockSize <= 0) {
                       ORT_LLM_LOG_WARNING(
                           "Invalid value for TRTLLM_MMHA_KERNEL_BLOCK_SIZE. Will use default values instead!");
                     }
                   }
                 });
  return mmhaKernelBlockSize;
}

bool getEnvUseTileSizeKv64ForTrtllmGen() {
  static bool const useTileSizeKv64 = getBoolEnv("TRTLLM_GEN_ENABLE_TILE_SIZE_KV64");
  return useTileSizeKv64;
}

bool getEnvEnablePDL() {
  static std::once_flag flag;
  static bool enablePDL = false;

  std::call_once(flag,
                 [&]() {
                   if (getSMVersion() >= 90) {
                     // PDL will be enabled by setting the env variables `TRTLLM_ENABLE_PDL` to `1`
                     enablePDL = getBoolEnv("TRTLLM_ENABLE_PDL");
                   }
                 });
  return enablePDL;
}

bool getEnvUseUCXKvCache() {
  static bool const useUCXKVCache = getBoolEnv("TRTLLM_USE_UCX_KVCACHE");
  return useUCXKVCache;
}

bool getEnvUseMPIKvCache() {
  static bool const useMPIKVCache = getBoolEnv("TRTLLM_USE_MPI_KVCACHE");
  return useMPIKVCache;
}

bool getEnvUseNixlKvCache() {
  static bool const useNixlKvCache = getBoolEnv("TRTLLM_USE_NIXL_KVCACHE");
  return useNixlKvCache;
}

std::string getEnvUCXInterface() {
  static std::once_flag flag;
  static std::string ucxInterface;

  std::call_once(flag,
                 [&]() {
                   char const* ucx_interface = std::getenv("TRTLLM_UCX_INTERFACE");
                   if (ucx_interface) {
                     ucxInterface = ucx_interface;
                   }
                 });
  return ucxInterface;
}

std::string getEnvNixlInterface() {
  static std::once_flag flag;
  static std::string nixlInterface;

  std::call_once(flag,
                 [&]() {
                   char const* nixl_interface = std::getenv("TRTLLM_NIXL_INTERFACE");
                   if (nixl_interface) {
                     nixlInterface = nixl_interface;
                   }
                 });
  return nixlInterface;
}

bool getEnvDisaggLayerwise() {
  static bool const disaggLayerwise = getBoolEnv("TRTLLM_DISAGG_LAYERWISE");
  return disaggLayerwise;
}

bool getEnvDisableSelectiveCacheTransfer() {
  static bool const disableSelectiveCacheTransfer = getBoolEnv("TRTLLM_DISABLE_SELECTIVE_CACHE_TRANSFER");
  return disableSelectiveCacheTransfer;
}

bool getEnvParallelCacheSend() {
  static bool const parallelCacheSend = getBoolEnv("TRTLLM_PARALLEL_CACHE_SEND");
  return parallelCacheSend;
}

bool getEnvRequestKVCacheConcurrent() {
  static bool const requestKVCacheConcurrent = getBoolEnv("TRTLLM_REQUEST_KV_CACHE_CONCURRENT");
  return requestKVCacheConcurrent;
}

bool getEnvDisableKVCacheTransferOverlap() {
  static bool const disableKVCacheTransferOverlap = getBoolEnv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP");
  return disableKVCacheTransferOverlap;
}

bool getEnvEnableReceiveKVCacheParallel() {
  static bool const enableReceiveParallel = getBoolEnv("TRTLLM_ENABLE_KVCACHE_RECEIVE_PARALLEL");
  return enableReceiveParallel;
}

bool getEnvTryZCopyForKVCacheTransfer() {
  static bool const zcopyForSysmmetricKVCache = getBoolEnv("TRTLLM_TRY_ZCOPY_FOR_KVCACHE_TRANSFER");
  return zcopyForSysmmetricKVCache;
}

bool getEnvForceDeterministic() {
  static bool const forceDeterministic = getBoolEnv("FORCE_DETERMINISTIC");
  return forceDeterministic;
}

bool getEnvForceDeterministicMOE() {
  static bool const forceDeterministic = getBoolEnv("FORCE_MOE_KERNEL_DETERMINISTIC") || getEnvForceDeterministic();
  return forceDeterministic;
}

bool getEnvForceDeterministicAttention() {
  static bool const forceDeterministic = getBoolEnv("FORCE_ATTENTION_KERNEL_DETERMINISTIC") || getEnvForceDeterministic();
  return forceDeterministic;
}

bool getEnvForceDeterministicAllReduce() {
  static bool const forceDeterministic = getBoolEnv("FORCE_ALL_REDUCE_DETERMINISTIC") || getEnvForceDeterministic();
  return forceDeterministic;
}

size_t getEnvAllReduceWorkspaceSize() {
  static size_t const workspaceSize = getUInt64Env("FORCE_ALLREDUCE_KERNEL_WORKSPACE_SIZE").value_or(1000 * 1000 * 1000);
  return workspaceSize;
}

std::string getEnvKVCacheTransferOutputPath() {
  static std::string outputPath = getStrEnv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH").value_or("");
  return outputPath;
}

bool getEnvKVCacheTransferUseAsyncBuffer() {
  static bool const useAsyncBuffer = getBoolEnv("TRTLLM_KVCACHE_TRANSFER_USE_ASYNC_BUFFER");
  return useAsyncBuffer;
}

bool getEnvKVCacheTransferUseSyncBuffer() {
  static bool const useSyncBuffer = getBoolEnv("TRTLLM_KVCACHE_TRANSFER_USE_SYNC_BUFFER");
  return useSyncBuffer;
}

size_t getEnvKVCacheSendMaxConcurrenceNum() {
  static size_t const maxConcurrenceNum = getUInt64Env("TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM").value_or(2);
  return maxConcurrenceNum;
}

size_t getEnvKVCacheRecvBufferCount() {
  static size_t const recvBufferCount = getUInt64Env("TRTLLM_KVCACHE_RECV_BUFFER_COUNT").value_or(2);
  return recvBufferCount;
}

size_t getEnvMemSizeForKVCacheTransferBuffer() {
  static std::once_flag flag;
  static size_t memSizeForKVCacheTransferBuffer = 0;

  std::call_once(flag,
                 [&]() {
                   char const* memSizeForKVCacheTransferBufferEnv = std::getenv("TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE");
                   if (memSizeForKVCacheTransferBufferEnv) {
                     memSizeForKVCacheTransferBuffer = parseMemorySize(memSizeForKVCacheTransferBufferEnv);
                   } else {
                     memSizeForKVCacheTransferBuffer = parseMemorySize("512MB");
                   }
                 });

  return memSizeForKVCacheTransferBuffer;
}

uint16_t getEnvNixlPort() {
  static uint16_t const nixlPort = getUInt64Env("TRTLLM_NIXL_PORT").value_or(0);
  return nixlPort;
}

bool getEnvDisaggBenchmarkGenOnly() {
  return getBoolEnv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY");
}

bool getEnvDisableChunkedAttentionInGenPhase() {
  return getBoolEnv("TRTLLM_DISABLE_CHUNKED_ATTENTION_IN_GEN_PHASE");
}

}  // namespace onnxruntime::llm::common
