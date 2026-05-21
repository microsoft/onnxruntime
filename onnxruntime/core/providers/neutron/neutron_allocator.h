// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <stdint.h>
#include <vector>

#include "core/common/common.h"

#define NEUTRON_DEFAULT_512MB_SLOTS 4
#define NEUTRON_MAX_512MB_SLOTS 6

namespace onnxruntime {

constexpr size_t kDefaultTensorAlignment = 64;
constexpr size_t kBoundaryNeutronBufferSize = 512 * 1024 * 1024;
constexpr size_t kReservedNeutronBufferSize = 192 * 1024 * 1024;
constexpr size_t kMaxNeutronNumHandles = NEUTRON_MAX_512MB_SLOTS;

class NeutronStackAllocator {
 public:
  NeutronStackAllocator() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NeutronStackAllocator);

  bool Init();

  /// Returns the memory slot handle with the most free space.
  /// Call this before Alloc() to select the optimal slot.
  size_t getMemoryHandle();

  /// Allocate memory from the specified handle slot.
  /// Uses reserved space (192MB) + requested size.
  void* Alloc(size_t size, size_t handle);

  /// Allocate from the remaining free space (no reserved buffer).
  void* AllocReserved(size_t size, size_t handle);

  /// Save current allocation state for later rollback.
  void pushMemoryState(size_t handle);

  /// Release all allocations since the last push and restore state.
  void popMemoryState(size_t handle);

  ~NeutronStackAllocator();

 private:
  uint8_t* p_{NULL};
  size_t neutronNumHandles{NEUTRON_DEFAULT_512MB_SLOTS};
  uint8_t* neutron_ptr_[kMaxNeutronNumHandles];
  size_t neutron_size_[kMaxNeutronNumHandles];
  std::vector<uint8_t*> past_ptrs_;
  std::vector<size_t> past_sizes_;
};

}  // namespace onnxruntime
