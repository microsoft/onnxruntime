// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "gsl/gsl"

struct OrtKeyValuePairs {
  OrtKeyValuePairs() = default;

  OrtKeyValuePairs(const OrtKeyValuePairs& other) {
    CopyFromMap(other.entries_);
  }

  OrtKeyValuePairs(OrtKeyValuePairs&& other) noexcept : OrtKeyValuePairs{} {
    swap(*this, other);
  }

  OrtKeyValuePairs& operator=(OrtKeyValuePairs other) noexcept {  // handles copy and move assignment
    swap(*this, other);
    return *this;
  }

  friend void swap(OrtKeyValuePairs& a, OrtKeyValuePairs& b) {
    using std::swap;
    swap(a.entries_, b.entries_);
    swap(a.keys_, b.keys_);
    swap(a.values_, b.values_);
  }

  void CopyFromMap(std::map<std::string, std::string> src) {
    entries_ = std::move(src);
    Sync();
  }

  void Add(const char* key, const char* value) {
    // ignore if either are nullptr.
    if (key && value) {
      Add(std::string(key), std::string(value));
    }
  }

  void Add(std::string key, std::string value) {
    if (key.empty()) {  // ignore empty keys
      return;
    }

    auto [it, inserted] = entries_.insert_or_assign(std::move(key), std::move(value));
    if (inserted) {
      const auto& [entry_key, entry_value] = *it;
      keys_.push_back(entry_key.c_str());
      values_.push_back(entry_value.c_str());
    } else {
      // rebuild is easier and changing an entry is not expected to be a common case.
      Sync();
    }
  }

  // we don't expect this to be common. reconsider using std::vector if it turns out to be.
  void Remove(const char* key) {
    if (key == nullptr) {
      return;
    }

    auto iter = entries_.find(key);
    if (iter != entries_.end()) {
      auto key_iter = std::find(keys_.begin(), keys_.end(), iter->first.c_str());
      // there should only ever be one matching entry, and keys_ and values_ should be in sync
      if (key_iter != keys_.end()) {
        auto idx = std::distance(keys_.begin(), key_iter);
        keys_.erase(key_iter);
        values_.erase(values_.begin() + idx);
      }

      entries_.erase(iter);
    }
  }

  const std::map<std::string, std::string>& Entries() const {
    return entries_;
  }

  gsl::span<const char* const> Keys() const {
    return keys_;
  }

  gsl::span<const char* const> Values() const {
    return values_;
  }

 private:
  void Sync() {
    keys_.clear();
    values_.clear();
    for (const auto& entry : entries_) {
      keys_.push_back(entry.first.c_str());
      values_.push_back(entry.second.c_str());
    }
  }

  // Note: Use std::map so that we can iterate through entries in a deterministic order.
  std::map<std::string, std::string> entries_;

  // members to make returning all key/value entries via the C API easier
  // Note: The elements point to strings owned by `entries_`.
  std::vector<const char*> keys_;
  std::vector<const char*> values_;
};
