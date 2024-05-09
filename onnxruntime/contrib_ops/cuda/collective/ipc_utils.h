
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

Status SetPeerAccess(int rank_id, int n_ranks, bool enable = true);

class IpcMemory {
 public:
  size_t static constexpr FLAGS_SIZE = (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t);

  IpcMemory(int rank_id, int n_ranks, std::size_t buffer_size);
  ~IpcMemory();

  std::vector<void*> const& GetCommPtrsTensor() const {
    return m_comm_ptrs_;
  }

 private:
  Status AllocateIpcMemory();
  Status DestroyIpcMemory();

  int rank_id_;
  int n_ranks_;
  std::vector<void*> m_comm_ptrs_;
  std::size_t mbuffer_size_;
  void* m_buffer_ptr_{nullptr};
};

Status GetCustomAllReduceWorkspace(int rank_id, int n_ranks, size_t input_size,
                                   std::vector<std::unique_ptr<IpcMemory>>& m_ipc_momery_handles,
                                   std::vector<const void*>& m_comm_ptrs);

}  // namespace ort_trtllm
