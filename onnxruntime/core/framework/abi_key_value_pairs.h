// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>

struct OrtKeyValuePairs {
  std::unordered_map<std::string, std::string> entries;
  // members to make returning all key/value entries via the C API easier
  std::vector<const char*> keys;
  std::vector<const char*> values;

  void Copy(const std::unordered_map<std::string, std::string>& src) {
    entries = src;
    Sync();
  }
  void Add(const char* key, const char* value) {
    // ignore if either are nullptr.
    if (key && value) {
      Add(std::string(key), std::string(value));
    }
  }

  void Add(const std::string& key, const std::string& value) {
    if (key.empty()) {  // ignore empty keys
      return;
    }

    auto iter_inserted = entries.insert({key, value});
    bool inserted = iter_inserted.second;
    if (inserted) {
      const auto& entry = *iter_inserted.first;
      keys.push_back(entry.first.c_str());
      values.push_back(entry.second.c_str());
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

    auto iter = entries.find(key);
    if (iter != entries.end()) {
      auto key_iter = std::find(keys.begin(), keys.end(), iter->first.c_str());
      // there should only ever be one matching entry, and keys and values should be in sync
      if (key_iter != keys.end()) {
        auto idx = std::distance(keys.begin(), key_iter);
        keys.erase(key_iter);
        values.erase(values.begin() + idx);
      }

      entries.erase(iter);
    }
  }

 private:
  void Sync() {
    keys.clear();
    values.clear();
    for (const auto& entry : entries) {
      keys.push_back(entry.first.c_str());
      values.push_back(entry.second.c_str());
    }
  }
};
