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

#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace onnxruntime::llm::common {
// Useful when you want to inject some debug code controllable with env var.
std::optional<int32_t> getIntEnv(char const* name);

std::optional<size_t> getUInt64Env(char const* name);

bool getBoolEnv(char const* name);

// XQA kernels (optimized kernels for generation phase).
bool forceXQAKernels();

// Whether XQA JIT is enabled.
//
// Returns the value of TRTLLM_ENABLE_XQA_JIT env var. If such env var doesn't exist, std::nullopt is returned.
std::optional<bool> getEnvEnableXQAJIT();

// 0 means to use heuristics.
std::optional<int32_t> getEnvXqaBlocksPerSequence();

// Whether use tileSizeKv64 for multiCtasKvMode of trtllm-gen kernels.
bool getEnvUseTileSizeKv64ForTrtllmGen();

// Tune the number of blocks per sequence for accuracy/performance purpose.
bool getEnvMmhaMultiblockDebug();

int getEnvMmhaBlocksPerSequence();

int getEnvMmhaKernelBlockSize();

// Whether PDL is enabled.
bool getEnvEnablePDL();

bool getEnvUseUCXKvCache();

bool getEnvUseMPIKvCache();
bool getEnvUseNixlKvCache();

std::string getEnvUCXInterface();

std::string getEnvNixlInterface();

bool getEnvDisaggLayerwise();

bool getEnvDisableSelectiveCacheTransfer();

bool getEnvParallelCacheSend();

bool getEnvRequestKVCacheConcurrent();

bool getEnvDisableKVCacheTransferOverlap();

bool getEnvEnableReceiveKVCacheParallel();

std::string getEnvKVCacheTransferOutputPath();

bool getEnvTryZCopyForKVCacheTransfer();

// Force deterministic behavior for all kernels.
bool getEnvForceDeterministic();

// Force deterministic behavior for MoE plugin.
bool getEnvForceDeterministicMOE();

// Force deterministic behavior for attention plugin.
bool getEnvForceDeterministicAttention();

// Force deterministic behavior for all reduce plugin.
bool getEnvForceDeterministicAllReduce();

// Return the workspace size for custom all reduce kernels.
// This only works when force deterministic is enabled.
size_t getEnvAllReduceWorkspaceSize();

size_t getEnvKVCacheRecvBufferCount();

bool getEnvKVCacheTransferUseAsyncBuffer();

bool getEnvKVCacheTransferUseSyncBuffer();

size_t getEnvKVCacheSendMaxConcurrenceNum();

size_t getEnvMemSizeForKVCacheTransferBuffer();

uint16_t getEnvNixlPort();

bool getEnvDisaggBenchmarkGenOnly();

// Whether to disable the chunked-attention in the generation phase.
bool getEnvDisableChunkedAttentionInGenPhase();

}  // namespace onnxruntime::llm::common
