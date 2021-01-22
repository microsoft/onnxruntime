// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/allocation_planner.h"
#include <fstream>

namespace onnxruntime {
struct MemoryBlock {
  size_t offset_{0};
  size_t size_{0};

  MemoryBlock() = default;
  MemoryBlock(size_t offset, size_t size) : offset_(offset), size_(size) {}
};

class MemoryPattern {
  friend class MemPatternPlanner;

 public:
  MemoryPattern() = default;

  MemoryPattern(MemoryPattern&& rhs) noexcept
      : patterns_{std::move(rhs.patterns_)},
        peak_size_{std::move(rhs.peak_size_)} {}

  MemoryPattern& operator=(MemoryPattern&& rhs) noexcept {
    patterns_ = std::move(rhs.patterns_);
    peak_size_ = std::move(rhs.peak_size_);
    return *this;
  }

  size_t PeakSize() const {
    return peak_size_;
  }

  const MemoryBlock* GetBlock(int ml_value_idx) const {
    auto it = patterns_.find(ml_value_idx);
    if (it == patterns_.end())
      return nullptr;

    return &it->second;
  }

  // REVIEW (codemzs): Put some mechanism in place to ensure the sanity of the pattern when it is
  // amended, i.e integrity of peak size.
  void InsertBlock(int ml_value_idx, MemoryBlock block) {
    ORT_ENFORCE(patterns_.find(ml_value_idx) == patterns_.end());

    patterns_[ml_value_idx] = block;
  }

  void EraseBlock(int ml_value_idx) {
    ORT_ENFORCE(patterns_.find(ml_value_idx) != patterns_.end());

    patterns_.erase(ml_value_idx);
  }

  // allow move
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(MemoryPattern);
  std::unordered_map<int, MemoryBlock> patterns_;
  size_t peak_size_{0};
};

struct MemoryPatternGroup {
  std::vector<OrtMemoryInfo> locations;
  std::vector<MemoryPattern> patterns;

  const MemoryPattern* GetPatterns(const OrtMemoryInfo& location) const {
    for (size_t i = 0; i < locations.size(); i++)
      if (locations[i] == location) {
        return &patterns[i];
      }
    return nullptr;
  }

  void Serialize(std::ofstream& stream) {
    auto locations_size = locations.size();
    stream.write(reinterpret_cast<char*>(&locations_size), sizeof(locations_size));
    for (size_t index = 0; index < locations_size; index += 1) {
      size_t len = strlen(locations[index].name);
      stream.write(reinterpret_cast<char*>(&len), sizeof(len));
      stream.write(const_cast<char*>(locations[index].name), len);
      stream.write(reinterpret_cast<char*>(&(locations[index].id)), sizeof(locations[index].id));
      stream.write(reinterpret_cast<char*>(&(locations[index].mem_type)), sizeof(locations[index].mem_type));
      stream.write(reinterpret_cast<char*>(&(locations[index].alloc_type)), sizeof(locations[index].alloc_type));
      stream.write(reinterpret_cast<char*>(&(locations[index].device)), sizeof(locations[index].device));
    }

    auto patterns_size = patterns.size();
    stream.write(reinterpret_cast<char*>(&patterns_size), sizeof(patterns.size()));
    for (size_t index = 0; index < patterns.size(); index += 1) {
      auto internal_patterns_size = patterns[index].patterns_.size();
      stream.write(reinterpret_cast<char*>(&internal_patterns_size), sizeof(internal_patterns_size));
      for (auto it = patterns[index].patterns_.begin(); it != patterns[index].patterns_.end(); it++) {
        stream.write(reinterpret_cast<char*>(const_cast<int*>(&(it->first))), sizeof(int));
        stream.write(reinterpret_cast<char*>(&(it->second)), sizeof(MemoryBlock));
      }
      stream.write(reinterpret_cast<char*>(&(patterns[index].peak_size_)), sizeof(size_t));
    }
  }

  void Deserialize(std::ifstream& stream) {
    auto location_size = locations.size();
    stream.read(reinterpret_cast<char*>(&(location_size)), sizeof(location_size));
    auto offset = locations.size();
    for (size_t index = offset; index < location_size + offset; index += 1) {
      OrtMemoryInfo mem_info;
      locations.emplace_back(mem_info);
      size_t len;
      stream.read(reinterpret_cast<char*>(&len), sizeof(len));
      locations[index].name = (const char*)malloc((len + 1) * sizeof(char));
      stream.read(const_cast<char*>(locations[index].name), len);
      const_cast<char*>(locations[index].name)[len] = '\0';
      stream.read(reinterpret_cast<char*>(&(locations[index].id)), sizeof(locations[index].id));
      stream.read(reinterpret_cast<char*>(&(locations[index].mem_type)), sizeof(locations[index].mem_type));
      stream.read(reinterpret_cast<char*>(&(locations[index].alloc_type)), sizeof(locations[index].alloc_type));
      stream.read(reinterpret_cast<char*>(&(locations[index].device)), sizeof(locations[index].device));
    }

    auto pattern_size = patterns.size();
    stream.read(reinterpret_cast<char*>(&(pattern_size)), sizeof(pattern_size));
    offset = patterns.size();
    for (size_t index = offset; index < pattern_size + offset; index += 1) {
      MemoryPattern mem_pattern;
      patterns.emplace_back(std::move(mem_pattern));
      size_t patterns_size;
      stream.read(reinterpret_cast<char*>(&(patterns_size)), sizeof(patterns_size));
      for (size_t index_local = 0; index_local < patterns_size; index_local += 1) {
        int key;
        MemoryBlock mem_block;
        stream.read(reinterpret_cast<char*>(&key), sizeof(int));
        stream.read(reinterpret_cast<char*>(&mem_block), sizeof(MemoryBlock));
        patterns[index].patterns_[key] = mem_block;
      }
      stream.read(reinterpret_cast<char*>(&(patterns[index].peak_size_)), sizeof(size_t));
    }
  }
};
}  // namespace onnxruntime
