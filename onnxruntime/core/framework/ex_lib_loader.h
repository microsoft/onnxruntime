// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <memory>
#include <map>
#include "core/common/common.h"
#include "core/common/status.h"

namespace onnxruntime {
class ExLibLoader {
 public:
  ExLibLoader() = default;
  common::Status LoadExternalLib(const std::string& dso_file_path,
                                 void** handle);
  void* GetExLibHandle(const std::string& dso_file_path) const;

  virtual ~ExLibLoader();

 protected:
  virtual void PreUnloadLibrary(void* /*handle*/){};

  std::map<std::string, void*> dso_name_data_map_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExLibLoader);
};
}  // namespace onnxruntime
