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
#include "mpi_include.h"

namespace ort_trtllm {

using namespace onnxruntime;

Status SetPeerAccess(int rank, int world_size, bool enable) {
  const int src_node = rank;

  for (int dst_node = 0; dst_node < world_size; dst_node++) {
    if (dst_node == src_node) {
      continue;
    }

    int can_access_peer;
    CUDA_RETURN_IF_ERROR(cudaDeviceCanAccessPeer(&can_access_peer, src_node, dst_node));

    if (enable) {
      cudaDeviceEnablePeerAccess(dst_node, 0);
    } else {
      cudaDeviceDisablePeerAccess(dst_node);
    }
    auto const error = cudaGetLastError();
    if (error != cudaErrorPeerAccessAlreadyEnabled && error != cudaErrorPeerAccessNotEnabled) {
      CUDA_RETURN_IF_ERROR(error);
    }
  }

  return Status::OK();
}

IpcMemory::IpcMemory(int rank, int world_size, std::size_t buffer_size)
    : rank_(rank), world_size_(world_size), m_comm_ptrs_(world_size), mbuffer_size_(buffer_size) {
  ORT_ENFORCE(AllocateIpcMemory() == Status::OK());
}

Status IpcMemory::AllocateIpcMemory() {
  CUDA_RETURN_IF_ERROR(cudaMalloc(&m_buffer_ptr_, mbuffer_size_));
  CUDA_RETURN_IF_ERROR(cudaMemset(m_buffer_ptr_, 0, mbuffer_size_));

  cudaIpcMemHandle_t local_handle;
  CUDA_RETURN_IF_ERROR(cudaIpcGetMemHandle(&local_handle, m_buffer_ptr_));

  // Assume no pipeline parallelism.
  std::vector<char> serial_handles(CUDA_IPC_HANDLE_SIZE * world_size_, 0);

#ifdef USE_MPI
  MPI_CHECK(MPI_Allgather(local_handle.reserved, CUDA_IPC_HANDLE_SIZE, MPI_BYTE, serial_handles.data(),
                          CUDA_IPC_HANDLE_SIZE, MPI_BYTE, MPI_COMM_WORLD));
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Please compile ORT with USE_MPI.");
#endif

  std::vector<cudaIpcMemHandle_t> handles(world_size_);
  for (size_t i = 0; i < handles.size(); ++i) {
    memcpy(handles[i].reserved, &serial_handles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
  }

  for (size_t node_id = 0; node_id < handles.size(); node_id++) {
    if ((int)node_id == rank_) {
      m_comm_ptrs_[node_id] = m_buffer_ptr_;
    } else {
      uint8_t* foreign_buffer;
      CUDA_RETURN_IF_ERROR(cudaIpcOpenMemHandle(
          reinterpret_cast<void**>(&foreign_buffer), handles[node_id], cudaIpcMemLazyEnablePeerAccess));
      m_comm_ptrs_[node_id] = foreign_buffer;
    }
  }

  return Status::OK();
}

IpcMemory::~IpcMemory() {
  std::ignore = DestroyIpcMemory();
}

Status IpcMemory::DestroyIpcMemory() {
  for (int node_id = 0; node_id < world_size_; ++node_id) {
    if (node_id == rank_) {
      CUDA_RETURN_IF_ERROR(cudaFree(m_comm_ptrs_[node_id]));
    } else {
      CUDA_RETURN_IF_ERROR(cudaIpcCloseMemHandle(m_comm_ptrs_[node_id]));
    }
  }
  cudaFree(m_buffer_ptr_);
  return Status::OK();
}

Status GetCustomAllReduceWorkspace(int rank, int world_size, size_t input_size,
                                   std::vector<std::unique_ptr<IpcMemory>>& m_ipc_memory_handles,
                                   std::vector<const void*>& m_comm_ptrs) {
  ORT_ENFORCE(ort_trtllm::SetPeerAccess(rank, world_size, true) == Status::OK());
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  const std::size_t buffer_size = world_size * input_size;

  m_ipc_memory_handles.clear();
  m_ipc_memory_handles.emplace_back(std::make_unique<ort_trtllm::IpcMemory>(rank, world_size, buffer_size));
  m_ipc_memory_handles.emplace_back(
      std::make_unique<ort_trtllm::IpcMemory>(rank, world_size, ort_trtllm::IpcMemory::FLAGS_SIZE * world_size));
  m_ipc_memory_handles.emplace_back(
      std::make_unique<ort_trtllm::IpcMemory>(rank, world_size, ort_trtllm::IpcMemory::FLAGS_SIZE * world_size));
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  m_comm_ptrs.reserve(m_ipc_memory_handles.size() * world_size);
  m_comm_ptrs.resize(m_ipc_memory_handles.size() * world_size);

  for (size_t mem_idx = 0; mem_idx < m_ipc_memory_handles.size(); mem_idx++) {
    auto const& mem_comm_ptrs = m_ipc_memory_handles[mem_idx]->GetCommPtrsTensor();
    for (size_t tpIdx = 0; tpIdx < static_cast<size_t>(world_size); tpIdx++) {
      m_comm_ptrs[mem_idx * world_size + tpIdx] = mem_comm_ptrs[tpIdx];
    }
  }

  return Status::OK();
}

}  // namespace ort_trtllm
