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

namespace onnxruntime {
namespace cuda {
namespace collective {

#if defined(USE_MPI) || defined(ORT_USE_NCCL)

using namespace onnxruntime;

IpcMemory::IpcMemory(int rank, int world_size, std::size_t buffer_size)
    : rank_(rank), world_size_(world_size), m_comm_ptrs_(world_size), mbuffer_size_(buffer_size) {
  ORT_ENFORCE(AllocateIpcMemory() == Status::OK());
}

Status IpcMemory::AllocateIpcMemory() {
  void* m_buffer_ptr;
  CUDA_RETURN_IF_ERROR(cudaMalloc(&m_buffer_ptr, mbuffer_size_));
  m_buffer_uptr_ = CudaMemPtrT{m_buffer_ptr, CudaDeleter()};

  CUDA_RETURN_IF_ERROR(cudaMemset(m_buffer_uptr_.get(), 0, mbuffer_size_));

  cudaIpcMemHandle_t local_handle;
  CUDA_RETURN_IF_ERROR(cudaIpcGetMemHandle(&local_handle, m_buffer_uptr_.get()));

  // Assume no pipeline parallelism.
  InlinedVector<char> serial_handles(CUDA_IPC_HANDLE_SIZE * world_size_, 0);

#ifdef USE_MPI
  MPI_CHECK(MPI_Allgather(local_handle.reserved, CUDA_IPC_HANDLE_SIZE, MPI_BYTE, serial_handles.data(),
                          CUDA_IPC_HANDLE_SIZE, MPI_BYTE, MPI_COMM_WORLD));
#else
  // TODO: Implement this for NCCL.
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Please compile ORT with USE_MPI.");
#endif

  InlinedVector<cudaIpcMemHandle_t> handles(world_size_);
  for (size_t i = 0; i < handles.size(); ++i) {
    memcpy(handles[i].reserved, &serial_handles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
  }

  for (size_t node_id = 0; node_id < handles.size(); node_id++) {
    if ((int)node_id == rank_) {
      m_comm_ptrs_[node_id] = m_buffer_uptr_.get();
    } else {
      uint8_t* foreign_buffer;
      CUDA_RETURN_IF_ERROR(cudaIpcOpenMemHandle(
          reinterpret_cast<void**>(&foreign_buffer), handles[node_id], cudaIpcMemLazyEnablePeerAccess));
      m_ipc_uptrs_.emplace_back(foreign_buffer, IpcDeleter());
      m_comm_ptrs_[node_id] = foreign_buffer;
    }
  }

  return Status::OK();
}

IpcMemory::~IpcMemory() {
}

Status GetCustomAllReduceWorkspace(int rank, int world_size, size_t input_size,
                                   IPCMemoryResourcePack& ipc_mem_res_pack) {
  if (input_size <= ipc_mem_res_pack.max_input_size) {
    return Status::OK();
  }

  const std::size_t buffer_size = world_size * input_size;

  InlinedVector<std::unique_ptr<IpcMemory>>& m_ipc_memory_handles = ipc_mem_res_pack.m_ipc_momery_handles;
  const size_t handles_size{m_ipc_memory_handles.size()};
  constexpr size_t k_num_handles{3};

  m_ipc_memory_handles.emplace_back(std::make_unique<IpcMemory>(rank, world_size, buffer_size));
  m_ipc_memory_handles.emplace_back(
      std::make_unique<IpcMemory>(rank, world_size, IpcMemory::FLAGS_SIZE * world_size));
  m_ipc_memory_handles.emplace_back(
      std::make_unique<IpcMemory>(rank, world_size, IpcMemory::FLAGS_SIZE * world_size));
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  InlinedVector<const void*>& m_comm_ptrs = ipc_mem_res_pack.m_comm_ptrs;
  m_comm_ptrs.reserve(k_num_handles * world_size);
  m_comm_ptrs.resize(k_num_handles * world_size);

  for (size_t mem_idx = handles_size; mem_idx < m_ipc_memory_handles.size(); mem_idx++) {
    const auto& mem_comm_ptrs = m_ipc_memory_handles[mem_idx]->GetCommPtrsTensor();
    for (size_t tpIdx = 0; tpIdx < static_cast<size_t>(world_size); tpIdx++) {
      m_comm_ptrs[(mem_idx - handles_size) * world_size + tpIdx] = mem_comm_ptrs[tpIdx];
    }
  }

  ipc_mem_res_pack.max_input_size = input_size;

  return Status::OK();
}

IPCMemoryResourcePack& IPCMemoryResourcePack::GetGlobalInstance() {
  static IPCMemoryResourcePack ipc_mem_res_pack;
  return ipc_mem_res_pack;
}

#endif

}  // namespace collective
}  // namespace cuda
}  // namespace onnxruntime
