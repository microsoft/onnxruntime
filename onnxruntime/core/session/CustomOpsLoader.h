// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <memory>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/customregistry.h"
#include "core/framework/custom_ops_author.h"

namespace onnxruntime {
class CustomOpsLoader final {
 public:
  CustomOpsLoader() = default;
  common::Status LoadCustomOps(const std::string& dso_file_path,
                               std::shared_ptr<CustomRegistry>& custom_registry);
  ~CustomOpsLoader();

 private:
  const std::string kGetAllKernelsSymbol = "GetAllKernels";
  const std::string kGetAllSchemasSymbol = "GetAllSchemas";
  const std::string kFreeKernelsContainerSymbol = "FreeKernelsContainer";
  const std::string kFreeSchemasContainerSymbol = "FreeSchemasContainer";

  struct DsoData {
    void* lib_handle = nullptr;
    KernelsContainer* kernels_container = nullptr;
    SchemasContainer* schemas_container = nullptr;
  };
  std::map<std::string, DsoData> dso_name_data_map_;

  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomOpsLoader);
};
}  // namespace onnxruntime
