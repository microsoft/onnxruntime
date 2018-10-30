// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <memory>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/customregistry.h"
#include "core/framework/custom_ops_author.h"
#include "core/framework/ex_lib_loader.h"

namespace onnxruntime {
class CustomOpsLoader final : public ExLibLoader {
 public:
  CustomOpsLoader() = default;
  common::Status LoadCustomOps(const std::string& dso_file_path,
                               std::shared_ptr<CustomRegistry>& custom_registry);

 protected:
  virtual void PreUnLoadHandle(void* handle) override;

 private:
  const std::string kGetAllKernelsSymbol = "GetAllKernels";
  const std::string kGetAllSchemasSymbol = "GetAllSchemas";
  const std::string kFreeKernelsContainerSymbol = "FreeKernelsContainer";
  const std::string kFreeSchemasContainerSymbol = "FreeSchemasContainer";

  struct DsoData {
    KernelsContainer* kernels_container = nullptr;
    SchemasContainer* schemas_container = nullptr;
  };
  std::map<void*, DsoData> dso_handle_data_map_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomOpsLoader);
};
}  // namespace onnxruntime
