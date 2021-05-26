// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <Python.h>

#ifndef SHARED_PROVIDER
#include "core/platform/env.h"
#endif

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

using AddressInfos = std::unordered_map<void*, std::vector<std::string>>;

class RefCountTracker {
 public:
  enum ObjCategory {
    PythonCallArgs,
    PythonCallResults,
    AutoGradContext,
  };

#ifndef SHARED_PROVIDER
  static RefCountTracker& GetInstance() {
    static RefCountTracker tracker;
    return tracker;
  }
#else
  static RefCountTracker& GetInstance() { return Provider_GetHost()->GetRefCountTrackerInstance(); }
#endif

  void TrackPyObject(RefCountTracker::ObjCategory category, PyObject* py_obj, const std::string& log_tag) const;
  void DumpDetails(const std::string& phase_name) const;
  void Reset() const;

 private:
  RefCountTracker() {
    addr_info_map_ = {
        {RefCountTracker::ObjCategory::PythonCallArgs, forward_arg_addresses_},
        {RefCountTracker::ObjCategory::PythonCallResults, return_value_addresses_},
        {RefCountTracker::ObjCategory::AutoGradContext, auto_grad_addresses_},
    };
  }

  const char* ObjCategoryToString(int enumVal) {
    static const char* enum_strings[] = {"PythonCallArgs", "PythonCallResults", "AutoGradContext"};
    return enum_strings[enumVal];
  }

  AddressInfos forward_arg_addresses_;
  AddressInfos return_value_addresses_;
  AddressInfos auto_grad_addresses_;
  std::unordered_map<RefCountTracker::ObjCategory, AddressInfos> addr_info_map_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RefCountTracker);
};
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
