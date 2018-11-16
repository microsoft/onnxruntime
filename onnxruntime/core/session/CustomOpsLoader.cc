// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/CustomOpsLoader.h"

#include "core/framework/custom_ops_author.h"
#include "core/platform/env.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/severity.h"
#include <vector>

using namespace ::onnxruntime::common;
using namespace ::onnxruntime::logging;

namespace onnxruntime {
CustomOpsLoader::~CustomOpsLoader() {
  using FreeKernelsContainerFn = void (*)(KernelsContainer*);
  using FreeSchemasContainerFn = void (*)(SchemasContainer*);

  try {
    for (auto& elem : dso_name_data_map_) {
      LOGS_DEFAULT(INFO) << "Unloading DSO " << elem.first;

      // free memory
      void* lib_handle = elem.second.lib_handle;
      if (!lib_handle) {
        continue;
      }

      // free the kernels container
      if (elem.second.kernels_container) {
        void* free_all_kernels_symbol_handle = nullptr;
        Env::Default().GetSymbolFromLibrary(lib_handle,
                                            kFreeKernelsContainerSymbol,
                                            &free_all_kernels_symbol_handle);
        if (!free_all_kernels_symbol_handle) {
          LOGS_DEFAULT(WARNING) << "Got nullptr for " + kFreeKernelsContainerSymbol + " for DSO " + elem.first;
        } else {
          FreeKernelsContainerFn free_all_kernels_fn = reinterpret_cast<FreeKernelsContainerFn>(free_all_kernels_symbol_handle);
          free_all_kernels_fn(elem.second.kernels_container);
        }
      }

      // free the schemas container
      if (elem.second.schemas_container) {
        void* free_all_schemas_symbol_handle = nullptr;
        Env::Default().GetSymbolFromLibrary(lib_handle,
                                            kFreeSchemasContainerSymbol,
                                            &free_all_schemas_symbol_handle);

        if (!free_all_schemas_symbol_handle) {
          LOGS_DEFAULT(WARNING) << "Got nullptr for " + kFreeSchemasContainerSymbol + " for DSO " + elem.first;
        } else {
          FreeSchemasContainerFn free_all_schemas_fn = reinterpret_cast<FreeSchemasContainerFn>(free_all_schemas_symbol_handle);
          free_all_schemas_fn(elem.second.schemas_container);
        }
      }

      // unload the DSO
      if (!Env::Default().UnloadDynamicLibrary(lib_handle).IsOK()) {
        LOGS_DEFAULT(WARNING) << "Failed to unload DSO: " << elem.first;
      }
    }
  } catch (std::exception& ex) {  // make sure exceptions don't leave the destructor
    LOGS_DEFAULT(WARNING) << "Caught exception while destructing CustomOpsLoader with message: " << ex.what();
  }
}

Status CustomOpsLoader::LoadCustomOps(const std::string& dso_file_path,
                                      std::shared_ptr<CustomRegistry>& custom_registry) {
  try {
    if (dso_name_data_map_.count(dso_file_path)) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "A dso with name " + dso_file_path + " has already been loaded.");
    }

    using GetAllKernelsFn = KernelsContainer* (*)();
    using GetAllSchemasFn = SchemasContainer* (*)();
    void* lib_handle = nullptr;
    ONNXRUNTIME_RETURN_IF_ERROR(Env::Default().LoadDynamicLibrary(dso_file_path, &lib_handle));
    dso_name_data_map_[dso_file_path].lib_handle = lib_handle;

    // get symbol for GetAllKernels
    void* get_all_kernels_symbol_handle = nullptr;
    ONNXRUNTIME_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(lib_handle,
                                                            kGetAllKernelsSymbol,
                                                            &get_all_kernels_symbol_handle));
    if (!get_all_kernels_symbol_handle) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                    "Got null handle for " + kGetAllKernelsSymbol + " for DSO " + dso_file_path);
    }

    GetAllKernelsFn get_all_kernels_fn = reinterpret_cast<GetAllKernelsFn>(get_all_kernels_symbol_handle);
    KernelsContainer* kernels_container = get_all_kernels_fn();
    if (!kernels_container) {
      LOGS_DEFAULT(WARNING) << "Got nullptr for KernelsContainer from the custom op library " << dso_file_path;
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Got nullptr for KernelsContainer from the custom op library " + dso_file_path);
    }
    dso_name_data_map_[dso_file_path].kernels_container = kernels_container;

    // register the kernels
    custom_registry.reset();
    custom_registry = std::make_shared<CustomRegistry>();

    for (auto& i : kernels_container->kernels_list) {
      ONNXRUNTIME_RETURN_IF_ERROR(custom_registry->RegisterCustomKernel(i));
    }

    // get symbol for GetAllSchemas
    void* get_all_schemas_symbol_handle = nullptr;
    ONNXRUNTIME_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(lib_handle,
                                                            kGetAllSchemasSymbol,
                                                            &get_all_schemas_symbol_handle));

    if (!get_all_schemas_symbol_handle) {  // a custom schema may not be registered
      return Status::OK();
    }

    GetAllSchemasFn get_all_schemas_fn = reinterpret_cast<GetAllSchemasFn>(get_all_schemas_symbol_handle);
    SchemasContainer* schemas_container = get_all_schemas_fn();
    if (!schemas_container) {
      LOGS_DEFAULT(WARNING) << "Got nullptr for SchemasContainer from the custom op library " << dso_file_path;
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Got nullptr for SchemasContainer from the custom op library " + dso_file_path);
    }
    dso_name_data_map_[dso_file_path].schemas_container = schemas_container;

    // register the schemas if present
    ONNXRUNTIME_RETURN_IF_ERROR(custom_registry->RegisterOpSet(schemas_container->schemas_list,
                                                       schemas_container->domain,
                                                       schemas_container->baseline_opset_version,
                                                       schemas_container->opset_version));
    return Status::OK();
  } catch (const std::exception& ex) {
    return Status(ONNXRUNTIME, FAIL, "Caught exception while loading custom ops with message: " + std::string(ex.what()));
  }
}
}  // namespace onnxruntime
