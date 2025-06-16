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

#include <cuda_runtime_api.h>
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime::llm::common {
inline int getDevice() {
  int deviceID{0};
  CUDA_CALL_THROW(cudaGetDevice(&deviceID));
  return deviceID;
}

inline int getSMVersion() {
  int device{-1};
  CUDA_CALL_THROW(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

inline int getMultiProcessorCount() {
  int nSM{0};
  int deviceID{0};
  CUDA_CALL_THROW(cudaGetDevice(&deviceID));
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID));
  return nSM;
}
}  // namespace onnxruntime::llm::common
