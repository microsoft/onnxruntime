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

#include <functional>

#include "custom_reduce_impl.h"

namespace onnxruntime {
namespace cuda {
namespace collective {

#if defined(USE_MPI) || defined(ORT_USE_NCCL)

struct CudaDeleter {
  void operator()(void* ptr) const noexcept {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
};

struct IpcDeleter {
  void operator()(void* ptr) const noexcept {
    if (ptr != nullptr) {
      cudaIpcCloseMemHandle(ptr);
    }
  }
};

class IpcMemory {
 public:
  size_t static constexpr FLAGS_SIZE = (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t);

  // Exchanges `bytes` of opaque data from every rank, writing world_size * bytes into `recv` in
  // rank order. Supplied by the caller so this file does not have to know how the ranks found
  // each other: MPI_Allgather when ORT was launched by mpirun, ncclAllGather otherwise.
  using AllGatherFn = std::function<Status(const char* send, char* recv, size_t bytes)>;

  IpcMemory(int rank, int world_size, std::size_t buffer_size, const AllGatherFn& all_gather);
  ~IpcMemory();

  const InlinedVector<void*>& GetCommPtrsTensor() const {
    return m_comm_ptrs_;
  }

 private:
  Status AllocateIpcMemory(const AllGatherFn& all_gather);

  int rank_;
  int world_size_;
  InlinedVector<void*> m_comm_ptrs_;
  std::size_t mbuffer_size_;

  using CudaMemPtrT = std::unique_ptr<void, CudaDeleter>;
  CudaMemPtrT m_buffer_uptr_;

  using IpcMemPtrT = std::unique_ptr<void, IpcDeleter>;
  InlinedVector<IpcMemPtrT> m_ipc_uptrs_;
};

// A global resource pack for IPC memory used in custom reduce kernel.
// Resource retrieval and deserialization are made atomic to thread safety of accessing it.
struct IPCMemoryResourcePack {
  InlinedVector<std::unique_ptr<IpcMemory>> m_ipc_momery_handles;
  InlinedVector<const void*> m_comm_ptrs;
  size_t max_input_size{0};

  static IPCMemoryResourcePack& GetGlobalInstance();
};

Status
GetCustomAllReduceWorkspace(int rank, int world_size, size_t input_size, IPCMemoryResourcePack& ipc_mem_res_pack,
                            const IpcMemory::AllGatherFn& all_gather);

#endif

}  // namespace collective
}  // namespace cuda
}  // namespace onnxruntime
