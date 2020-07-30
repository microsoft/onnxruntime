// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <unordered_map>

// Thread-safe container wrapper for avoiding potential racing conditions
// when using parallel executor.
template <typename TKey, typename TValue>
class ThreadSafeUnorderedMap final {
 public:
  ThreadSafeUnorderedMap() {};
  ThreadSafeUnorderedMap(const ThreadSafeUnorderedMap& another) {
    // Get a fork of map_ from another instance.
    // For mutex, we use a new one initialized by its default ctor.
    std::unique_lock<std::mutex> lock(another.mtx_);
    map_ = another.map_;
  }

  ThreadSafeUnorderedMap& operator=(const ThreadSafeUnorderedMap& another) {
    // Get a fork of map_ from another instance.
    // For mutex, we use a new one initialized by its default ctor.
    std::unique_lock<std::mutex> lock(mtx_);
    std::unique_lock<std::mutex> another_lock(another.mtx_);
    map_ = another.map_;
    return *this;
  }

  void Set(const TKey& key, const TValue& value) {
    std::unique_lock<std::mutex> lock(mtx_);
    map_[key] = value;
  }

  bool TryFind(const TKey& key, TValue& value) const {
    std::unique_lock<std::mutex> lock(mtx_);
    auto it = map_.find(key);
    if (it != map_.end()) {
      value = it->second;
      return true;
    } else {
      return false;
    }
  }

 private:
  mutable std::mutex mtx_;
  std::unordered_map<TKey, TValue> map_;
};