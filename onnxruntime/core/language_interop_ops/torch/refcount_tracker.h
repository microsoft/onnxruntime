// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <Python.h>
#include "core/platform/env.h"

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {
#ifndef NDEBUG

using AddressInfos = std::unordered_map<void*, std::vector<std::string>>;

class RefCountTracker {
 public:
  enum ObjCategory {
    CallbackFunction,
    ForwardArgs
  };

  static RefCountTracker& GetInstance() {
    static RefCountTracker tracker;
    return tracker;
  }
  void TrackPyObject(RefCountTracker::ObjCategory category, PyObject* py_obj, std::string log_tag);
  void DumpDetails();
  void Reset();

 private:
  RefCountTracker();
  const char* ObjCategoryToString(int enumVal) {
    static const char* enum_strings[] = {"CallbackFunction", "ForwardArgs"};
    return enum_strings[enumVal];
  }
  AddressInfos func_addresses_;
  AddressInfos forward_arg_addresses_;
  std::unordered_map<RefCountTracker::ObjCategory, AddressInfos> addr_info_map_;
};
#endif
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
