// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/dlpack/python_common.h"

#include <iostream>
#include "core/common/logging/logging.h"
#include "core/language_interop_ops/torch/refcount_tracker.h"
#include "core/platform/env.h"

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

void RefCountTracker::TrackPyObject(RefCountTracker::ObjCategory category, PyObject* py_obj, const std::string& log_tag) {
#ifdef NDEBUG
  ORT_UNUSED_PARAMETER(category);
  ORT_UNUSED_PARAMETER(py_obj);
  ORT_UNUSED_PARAMETER(log_tag);
#else
  AddressInfos& addrs = addr_info_map_[category];
  assert(py_obj != NULL);
  void* addr = static_cast<void*>(py_obj);
  auto it = addrs.find(addr);
  if (it == addrs.end()) {
    addrs.insert({addr, {log_tag}});
  } else {
    addrs[addr].push_back(log_tag);
  }
  LOGS_DEFAULT(WARNING) << "Track" << ObjCategoryToString(category) << "\tAddress: [" << addr << "]\tRefCnt: " << Py_REFCNT(addr) << "\tLogTag: " << log_tag;
#endif
}

void RefCountTracker::DumpDetails(const std::string& phase_name) const {
#ifdef NDEBUG
  ORT_UNUSED_PARAMETER(phase_name);
#else
  std::ostringstream oss;
  oss << "======================" << phase_name << "=================" << std::endl;
  for (auto addr_info_it = addr_info_map_.begin(); addr_info_it != addr_info_map_.end(); ++addr_info_it) {
    oss << "Category: " << ObjCategoryToString(addr_info_it->first) << std::endl;
    for (auto it = addr_info_it->second.begin(); it != addr_info_it->second.end(); ++it) {
      oss << "\tAddress: [" << it->first << "] \tRefCnt: " << Py_REFCNT(it->first) << " \tLogTag: (";
      for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
        oss << *vit << ",";
      }
      oss << ")\n";
    }
  }
  oss << "==========================================================" << std::endl;
  LOGS_DEFAULT(WARNING) << oss.str();
#endif
}

void RefCountTracker::Reset() {
#ifndef NDEBUG
  for (auto addr_info_it = addr_info_map_.begin(); addr_info_it != addr_info_map_.end(); ++addr_info_it) {
    addr_info_it->second.clear();
  }
#endif
}

}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
