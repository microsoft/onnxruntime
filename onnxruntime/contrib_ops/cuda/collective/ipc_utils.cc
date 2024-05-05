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

#include "ipc_utils.h"

namespace ort_trtllm {

using namespace onnxruntime;

Status setPeerAccess(NcclContext* nctx, bool enable) {
  const int srcNode = nctx->Rank();

  for (int destNode = 0; destNode < nctx->Size(); destNode++) {
    if (destNode == srcNode) {
      continue;
    }

    int canAccessPeer;
    CUDA_RETURN_IF_ERROR(cudaDeviceCanAccessPeer(&canAccessPeer, srcNode, destNode));

    if (enable) {
      cudaDeviceEnablePeerAccess(destNode, 0);
    } else {
      cudaDeviceDisablePeerAccess(destNode);
    }
    auto const error = cudaGetLastError();
    if (error != cudaErrorPeerAccessAlreadyEnabled && error != cudaErrorPeerAccessNotEnabled) {
      CUDA_RETURN_IF_ERROR(error);
    }
  }

  return Status::OK();
}

IpcMemory::IpcMemory(NcclContext* nctx, std::size_t bufferSize)
    : nctx_(nctx), mCommPtrs(nctx_->Size()), mBufferSize(bufferSize) {
  ORT_ENFORCE(allocateIpcMemory() == Status::OK());
}

Status IpcMemory::allocateIpcMemory() {
  CUDA_RETURN_IF_ERROR(cudaMalloc(&mBufferPtr, mBufferSize));
  CUDA_RETURN_IF_ERROR(cudaMemset(mBufferPtr, 0, mBufferSize));

  cudaIpcMemHandle_t localHandle;
  CUDA_RETURN_IF_ERROR(cudaIpcGetMemHandle(&localHandle, mBufferPtr));

  // Assume no pipeline parallelism.
  std::vector<char> serialHandles(CUDA_IPC_HANDLE_SIZE * nctx_->Size(), 0);

  ncclComm_t comm = nctx_->Comm();
  NCCL_RETURN_IF_ERROR(
      ncclAllGather(&localHandle.reserved, serialHandles.data(), CUDA_IPC_HANDLE_SIZE, ncclUint8, comm, 0));

  std::vector<cudaIpcMemHandle_t> handles(nctx_->Size());
  for (size_t i = 0; i < handles.size(); ++i) {
    memcpy(handles[i].reserved, &serialHandles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
  }

  for (size_t nodeId = 0; nodeId < handles.size(); nodeId++) {
    if ((int)nodeId == nctx_->Rank()) {
      mCommPtrs[nodeId] = mBufferPtr;
    } else {
      uint8_t* foreignBuffer;
      CUDA_RETURN_IF_ERROR(cudaIpcOpenMemHandle(
          reinterpret_cast<void**>(&foreignBuffer), handles[nodeId], cudaIpcMemLazyEnablePeerAccess));
      mCommPtrs[nodeId] = foreignBuffer;
    }
  }
}

IpcMemory::~IpcMemory() {
  std::ignore = destroyIpcMemory();
}

Status IpcMemory::destroyIpcMemory() {
  for (int nodeId = 0; nodeId < nctx_->Size(); ++nodeId) {
    if (nodeId == nctx_->Rank()) {
      CUDA_RETURN_IF_ERROR(cudaFree(mCommPtrs[nodeId]));
    } else {
      CUDA_RETURN_IF_ERROR(cudaIpcCloseMemHandle(mCommPtrs[nodeId]));
    }
  }
  cudaFree(mBufferPtr);

  return Status::OK();
}

}  // namespace ort_trtllm
