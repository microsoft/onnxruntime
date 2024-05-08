
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

namespace ort_trtllm {

Status setPeerAccess(int myRank, int nRanks, bool enable = true);

class IpcMemory {
 public:
  // using TensorPtr = ITensor::SharedPtr;

  // MAX_ALL_REDUCE_BLOCKS for block_barrier, 1 for multi_gpu_barrier
  size_t static constexpr FLAGS_SIZE = (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t);

  IpcMemory(int myRank, int nRanks, std::size_t bufferSize);
  ~IpcMemory();

  std::vector<void*> const& getCommPtrsTensor() const {
    return mCommPtrs;
  }

 private:
  Status allocateIpcMemory();
  Status destroyIpcMemory();

  int myRank_;
  int nRanks_;
  std::vector<void*> mCommPtrs;
  std::size_t mBufferSize;
  void* mBufferPtr{nullptr};
};

// static std::unordered_map<int, std::vector<const void*>> mCommPtrsMap;
// static std::unordered_map<int, size_t> bufferSizeMap;
// static std::vector<const void*> getCustomAllReduceWorkspace(NcclContext* nctx, size_t input_size) {
//   const int myRank = nctx->Rank();

//   std::cout << "input_size: " << input_size << "bufferSizeMap[myRank]: " << bufferSizeMap[myRank] << std::endl;
//   if (input_size <= bufferSizeMap[myRank]) {
//     return mCommPtrsMap[myRank];
//   }
//   bufferSizeMap[myRank] = input_size;

//   ORT_ENFORCE(setPeerAccess(nctx, true) == Status::OK());

//   const int nRanks = nctx->Size();
//   const std::size_t bufferSize = nRanks * bufferSizeMap[myRank];

//   std::vector<std::shared_ptr<IpcMemory>> mIpcMemoryHandles;
//   mIpcMemoryHandles.emplace_back(std::make_shared<IpcMemory>(nctx, bufferSize));
//   mIpcMemoryHandles.emplace_back(
//       std::make_shared<IpcMemory>(nctx, IpcMemory::FLAGS_SIZE * nRanks));
//   mIpcMemoryHandles.emplace_back(
//       std::make_shared<IpcMemory>(nctx, IpcMemory::FLAGS_SIZE * nRanks));

//   mCommPtrsMap[myRank].resize(mIpcMemoryHandles.size() * nRanks);

//   for (size_t memIdx = 0; memIdx < mIpcMemoryHandles.size(); memIdx++) {
//     auto const& memCommPtrs = mIpcMemoryHandles[memIdx]->getCommPtrsTensor();
//     for (size_t tpIdx = 0; tpIdx < static_cast<size_t>(nRanks); tpIdx++) {
//       mCommPtrsMap[myRank][memIdx * nRanks + tpIdx] = memCommPtrs[tpIdx];
//     }
//   }

//   return mCommPtrsMap[myRank];
// }

// static std::vector<const void*> getCustomAllReduceWorkspace(NcclContext* nctx, size_t input_size) {
//   static std::vector<const void*> mCommPtrs;
//   static size_t bufferSizePerRank;
//   static std::vector<std::shared_ptr<IpcMemory>> mIpcMemoryHandles;

//   if (input_size <= bufferSizePerRank) {
//     return mCommPtrs;
//   }

//   ORT_ENFORCE(setPeerAccess(nctx, true) == Status::OK());

//   const int nRanks = nctx->Size();
//   const std::size_t bufferSize = nRanks * input_size;

//   mIpcMemoryHandles.clear();
//   mIpcMemoryHandles.emplace_back(std::make_shared<IpcMemory>(nctx, bufferSize));
//   mIpcMemoryHandles.emplace_back(
//       std::make_shared<IpcMemory>(nctx, IpcMemory::FLAGS_SIZE * nRanks));
//   mIpcMemoryHandles.emplace_back(
//       std::make_shared<IpcMemory>(nctx, IpcMemory::FLAGS_SIZE * nRanks));

//   mCommPtrs.resize(mIpcMemoryHandles.size() * nRanks);

//   for (size_t memIdx = 0; memIdx < mIpcMemoryHandles.size(); memIdx++) {
//     auto const& memCommPtrs = mIpcMemoryHandles[memIdx]->getCommPtrsTensor();
//     for (size_t tpIdx = 0; tpIdx < static_cast<size_t>(nRanks); tpIdx++) {
//       mCommPtrs[memIdx * nRanks + tpIdx] = memCommPtrs[tpIdx];
//     }
//   }

//   return mCommPtrs;
// }

}  // namespace ort_trtllm
