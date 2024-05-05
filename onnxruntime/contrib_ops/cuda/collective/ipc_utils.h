
/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "custom_reduce_impl.h"
#include "nccl_kernels.h"

namespace ort_trtllm {

using NcclContext = ::onnxruntime::contrib::cuda::NcclContext;

Status setPeerAccess(const NcclContext* nctx, bool enable = true);

class IpcMemory {
 public:
  // using TensorPtr = ITensor::SharedPtr;

  // MAX_ALL_REDUCE_BLOCKS for block_barrier, 1 for multi_gpu_barrier
  size_t static constexpr FLAGS_SIZE = (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t);

  IpcMemory(NcclContext* nctx, std::size_t bufferSize);
  ~IpcMemory();

  std::vector<void*> const& getCommPtrsTensor() const {
    return mCommPtrs;
  }

 private:
  Status allocateIpcMemory();
  Status destroyIpcMemory();

  NcclContext* nctx_;
  std::vector<void*> mCommPtrs;
  std::size_t mBufferSize;
  void* mBufferPtr{nullptr};
};

}  // namespace ort_trtllm
