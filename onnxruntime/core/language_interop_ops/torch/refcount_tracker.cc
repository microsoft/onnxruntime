// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Python.h>
#include <iostream>
#include "core/language_interop_ops/torch/refcount_tracker.h"
#include "core/platform/env.h"

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {
#ifndef NDEBUG

RefCountTracker::RefCountTracker() {
  addr_info_map_ = {
      {RefCountTracker::ObjCategory::ForwardArgs, forward_arg_addresses_},
      {RefCountTracker::ObjCategory::ReturnValues, return_value_addresses_},
  };
}

void RefCountTracker::TrackPyObject(RefCountTracker::ObjCategory category, PyObject* py_obj, std::string log_tag) {
  AddressInfos& addrs = addr_info_map_[category];
  void* addr = static_cast<void*>(py_obj);
  auto it = addrs.find(addr);
  if (it == addrs.end()) {
    addrs.insert({addr, {log_tag}});
  } else {
    addrs[addr].push_back(log_tag);
  }
  std::cout << "Track" << ObjCategoryToString(category) << "\tAddress: [" << addr << "]\tRefCnt: " << Py_REFCNT(addr) << "\tLogTag: " << log_tag << std::endl;
}

void RefCountTracker::DumpDetails() {
  for (auto addr_info_it = addr_info_map_.begin(); addr_info_it != addr_info_map_.end(); ++addr_info_it) {
    std::ostringstream oss;
    oss << "Category: " << ObjCategoryToString(addr_info_it->first) << std::endl;
    for (auto it = addr_info_it->second.begin(); it != addr_info_it->second.end(); ++it) {
      oss << "\tAddress: [" << it->first << "] \tRefCnt: " << Py_REFCNT(it->first) << " \tLogTag: (";
      for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
        oss << *vit << ",";
      }
      oss << ")\n";
    }
    std::cout << oss.str() << std::endl;
  }
}

void RefCountTracker::Reset() {
  for (auto addr_info_it = addr_info_map_.begin(); addr_info_it != addr_info_map_.end(); ++addr_info_it) {
    addr_info_it->second.clear();
  }
}

#endif
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
