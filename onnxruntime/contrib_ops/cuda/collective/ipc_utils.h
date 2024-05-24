
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

namespace onnxruntime {
namespace cuda {
namespace collective {

#if defined(USE_MPI) || defined(USE_NCCL)

class IpcMemory {
 public:
  size_t static constexpr FLAGS_SIZE = (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t);

  IpcMemory(int rank, int world_size, std::size_t buffer_size);
  ~IpcMemory();

  std::vector<void*> const& GetCommPtrsTensor() const {
    return m_comm_ptrs_;
  }

 private:
  Status DestroyIpcMemory();
  Status AllocateIpcMemory();

  int rank_;
  int world_size_;
  std::vector<void*> m_comm_ptrs_;
  std::size_t mbuffer_size_;
  void* m_buffer_ptr_{nullptr};
};

struct IPCMemoryResourcePack {
  mutable std::vector<std::shared_ptr<IpcMemory>> m_ipc_momery_handles;
  mutable std::vector<const void*> m_comm_ptrs;
  mutable size_t max_input_size{0};
  mutable uint32_t counter{0};
};

Status
GetCustomAllReduceWorkspace(int rank, int world_size, size_t input_size, IPCMemoryResourcePack& ipc_mem_res_pack);

class GlobalIPCMemoryResourcePack {
 public:
  IPCMemoryResourcePack& GetIPCMemoryResourcePack();
};

#endif

}  // namespace collective
}  // namespace cuda
}  // namespace onnxruntime
