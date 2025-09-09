// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

#include "core/common/hash_combine.h"
#include "core/framework/ortdevice.h"
#include "core/session/onnxruntime_c_api.h"  // for OrtMemType, OrtAllocatorType

struct OrtMemoryInfo {
  OrtMemoryInfo() = default;  // to allow default construction of Tensor

  // use string for name, so we could have customized allocator in execution provider.
  std::string name;
  OrtMemType mem_type = OrtMemTypeDefault;
  OrtAllocatorType alloc_type = OrtInvalidAllocator;
  OrtDevice device;

  OrtMemoryInfo(std::string name_, OrtAllocatorType type_, OrtDevice device_ = OrtDevice(),
                OrtMemType mem_type_ = OrtMemTypeDefault)
      : name(std::move(name_)),
        mem_type(mem_type_),
        alloc_type(type_),
        device(device_) {
  }

  // To make OrtMemoryInfo become a valid key in std map
  bool operator<(const OrtMemoryInfo& other) const {
    if (alloc_type != other.alloc_type)
      return alloc_type < other.alloc_type;
    if (mem_type != other.mem_type)
      return mem_type < other.mem_type;
    if (device != other.device)
      return device < other.device;

    return name < other.name;
  }

  // This is to make OrtMemoryInfo a valid key in hash tables
  size_t Hash() const {
    auto h = std::hash<int>()(alloc_type);
    onnxruntime::HashCombine(mem_type, h);
    onnxruntime::HashCombine(device.Hash(), h);
    onnxruntime::HashCombine<std::string_view>(name, h);
    return h;
  }

  std::string ToString() const {
    std::ostringstream ostr;
    ostr << "OrtMemoryInfo:["
         << "name:" << name
         << " OrtMemType:" << mem_type
         << " OrtAllocatorType:" << alloc_type
         << " " << device.ToString()
         << "]";
    return ostr.str();
  }
};

// Required by hash tables
inline bool operator==(const OrtMemoryInfo& left, const OrtMemoryInfo& other) {
  return left.mem_type == other.mem_type &&
         left.alloc_type == other.alloc_type &&
         left.device == other.device &&
         left.name == other.name;
}

inline bool operator!=(const OrtMemoryInfo& lhs, const OrtMemoryInfo& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream& out, const OrtMemoryInfo& info);

namespace std {
template <>
struct hash<OrtMemoryInfo> {
  size_t operator()(const OrtMemoryInfo& i) const {
    return i.Hash();
  }
};
}  // namespace std
