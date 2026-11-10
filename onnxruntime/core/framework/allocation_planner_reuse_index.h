// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <cstddef>
#include <functional>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

#include <gsl/span>

#include "core/common/common.h"
#include "core/common/safeint.h"

namespace onnxruntime::allocation_planner_internal {

// Planner-only full-word index. Head absence does not remove a key from a probe chain.
class FreeBufferIndex {
 public:
  FreeBufferIndex() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FreeBufferIndex);

  std::optional<size_t>* Find(size_t key) {
    if (slots_.empty()) return nullptr;
    auto& slot = slots_[FindSlot(slots_, key)];
    return slot.occupied ? &slot.head : nullptr;
  }

  std::optional<size_t>& GetOrCreate(size_t key) {
    if (slots_.empty()) Grow();
    size_t index = FindSlot(slots_, key);
    if (!slots_[index].occupied) {
      // Leave at least one unused slot, including when occupied heads are consumed.
      if (size_ == slots_.size() - slots_.size() / 4) {
        Grow();
        index = FindSlot(slots_, key);
      }
      auto& slot = slots_[index];
      slot.key = key;
      slot.head.reset();
      slot.occupied = true;
      ++size_;
    }
    return slots_[index].head;
  }

  void clear() {
    for (auto& slot : slots_) slot.occupied = false;
    size_ = 0;
  }

 private:
  friend struct FreeBufferIndexTestPeer;

  struct Slot {
    size_t key = 0;
    std::optional<size_t> head;
    bool occupied = false;
  };

  static size_t Hash(size_t key) {
    static_assert(std::has_unique_object_representations_v<size_t>);
    // Hash every byte to disperse aligned dimensions; equality still compares the full word.
    return std::hash<std::string_view>{}(
        std::string_view(reinterpret_cast<const char*>(&key), sizeof(key)));
  }

  static size_t FindSlot(gsl::span<const Slot> slots, size_t key) {
    const size_t mask = slots.size() - 1;
    size_t index = Hash(key) & mask;
    while (slots[index].occupied && slots[index].key != key) index = (index + 1) & mask;
    return index;
  }

  static size_t NextCapacity(size_t capacity) {
    if (capacity == 0) return 8;
    return SafeInt<size_t>(capacity) * 2;
  }

  void Grow() {
    std::vector<Slot> next(NextCapacity(slots_.size()));
    for (const auto& slot : slots_) {
      if (slot.occupied) next[FindSlot(next, slot.key)] = slot;
    }
    slots_.swap(next);
  }

  // A nonempty table exceeds inline-vector storage even at its initial capacity.
  std::vector<Slot> slots_;
  size_t size_ = 0;
};

}  // namespace onnxruntime::allocation_planner_internal

#endif  // !defined(ORT_MINIMAL_BUILD)
