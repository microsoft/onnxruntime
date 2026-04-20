// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#include <sys/mman.h>
#include <cstddef>

#include "core/providers/neutron/neutron_allocator.h"
#include "core/common/logging/logging.h"
#if NEUTRON_AARCH64
#include "neutron/NeutronDriver.h"
#endif

namespace onnxruntime {

inline size_t getAlignedSize(size_t size) {
  size_t mod = size % kDefaultTensorAlignment;
  return mod ? (size + kDefaultTensorAlignment - mod) : size;
}

NeutronStackAllocator::~NeutronStackAllocator() {
  if (p_)
    releaseBuffer(p_);
}

bool NeutronStackAllocator::Init() {
  static bool init = false;

  if (!init) {
    // Update number of 512MB slots from environment variable
    char* strNumHandles = NULL;
    strNumHandles = getenv("NEUTRON_CMA_512SLOTS");
    if (strNumHandles) {
      size_t num;
      if (sscanf(strNumHandles, "%ld", &num) == 1) {
        if (num >= 1 && num <= NEUTRON_MAX_512MB_SLOTS)
          neutronNumHandles = num;
      }
    }

    size_t fullNeutronBufferSize = neutronNumHandles * kBoundaryNeutronBufferSize;
    NeutronError ret = allocateBuffer(fullNeutronBufferSize, (void**)&p_, true);
    if (ret != ENONE)
      return false;

    LOGS_DEFAULT(WARNING) << "Alloc CMA userspace memory " << (fullNeutronBufferSize / 1024 / 1024) << " MB";
    for (size_t i = 0; i < neutronNumHandles; i++) {
      neutron_ptr_[i] = p_ + i * kBoundaryNeutronBufferSize;
      size_t rest = fullNeutronBufferSize - i * kBoundaryNeutronBufferSize;
      neutron_size_[i] = rest >= kBoundaryNeutronBufferSize ? kBoundaryNeutronBufferSize : (uint32_t)rest;

      LOGS_DEFAULT(WARNING) << "Alloc " << neutron_ptr_[i]
                            << (long)(neutron_size_[i] / 1024 / 1024)
                            << "MB for slot " << i;
    }

    init = true;
  }
  return true;
}

size_t NeutronStackAllocator::getMemoryHandle() {
  size_t largest_pos = 0;
  for (size_t i = 1; i < neutronNumHandles; i++)
    if (neutron_size_[i] > neutron_size_[largest_pos])
      largest_pos = i;
  return largest_pos;
}

void* NeutronStackAllocator::Alloc(size_t size, size_t handle) {
  if (p_ == NULL) {
    throw std::bad_alloc();
  }

  size = getAlignedSize(size);
  if (neutron_size_[handle] < (kReservedNeutronBufferSize + size)) {
    throw std::bad_alloc();
  }

  void* tmp = neutron_ptr_[handle];
  neutron_ptr_[handle] += size;
  neutron_size_[handle] -= size;

  return tmp;
}

void* NeutronStackAllocator::AllocReserved(size_t size, size_t handle) {
  size = getAlignedSize(size);
  if (neutron_size_[handle] < size) {
    throw std::bad_alloc();
  }

  void* tmp = neutron_ptr_[handle];
  neutron_ptr_[handle] += size;
  neutron_size_[handle] -= size;
  return tmp;
}

void NeutronStackAllocator::pushMemoryState(size_t handle) {
  past_ptrs_.push_back(neutron_ptr_[handle]);
  past_sizes_.push_back(neutron_size_[handle]);
}

void NeutronStackAllocator::popMemoryState(size_t handle) {
  neutron_ptr_[handle] = past_ptrs_.back();
  past_ptrs_.pop_back();
  neutron_size_[handle] = past_sizes_.back();
  past_sizes_.pop_back();
}

}  // namespace onnxruntime
