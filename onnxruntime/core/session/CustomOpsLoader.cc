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
void CustomOpsLoader::PreUnLoadHandle(void* handle) {
  using FreeKernelsContainerFn = void (*)(KernelsContainer*);
  using FreeSchemasContainerFn = void (*)(SchemasContainer*);
  auto it = dso_handle_data_map_.find(handle);
  if (it == dso_handle_data_map_.end())
    return;
  // free memory
  if (!handle)
    return;

  // free the kernels container
  if (it->second.kernels_container) {
    void* free_all_kernels_symbol_handle = nullptr;
    Env::Default().GetSymbolFromLibrary(handle,
                                        kFreeKernelsContainerSymbol,
                                        &free_all_kernels_symbol_handle);
    if (!free_all_kernels_symbol_handle) {
      LOGS_DEFAULT(WARNING) << "Got nullptr for " + kFreeKernelsContainerSymbol;
    } else {
      FreeKernelsContainerFn free_all_kernels_fn = reinterpret_cast<FreeKernelsContainerFn>(free_all_kernels_symbol_handle);
      free_all_kernels_fn(it->second.kernels_container);
    }
  }

  // free the schemas container
  if (it->second.schemas_container) {
    void* free_all_schemas_symbol_handle = nullptr;
    Env::Default().GetSymbolFromLibrary(handle,
                                        kFreeSchemasContainerSymbol,
                                        &free_all_schemas_symbol_handle);

    if (!free_all_schemas_symbol_handle) {
      LOGS_DEFAULT(WARNING) << "Got nullptr for " + kFreeSchemasContainerSymbol;
    } else {
      FreeSchemasContainerFn free_all_schemas_fn = reinterpret_cast<FreeSchemasContainerFn>(free_all_schemas_symbol_handle);
      free_all_schemas_fn(it->second.schemas_container);
    }
  }
}

Status CustomOpsLoader::LoadCustomOps(const std::string& dso_file_path,
                                      std::shared_ptr<CustomRegistry>& custom_registry) {
  void* lib_handle = nullptr;
  ORT_RETURN_IF_ERROR(LoadExternalLib(dso_file_path, &lib_handle));
  try {
    using GetAllKernelsFn = KernelsContainer* (*)();
    using GetAllSchemasFn = SchemasContainer* (*)();

    // get symbol for GetAllKernels
    void* get_all_kernels_symbol_handle = nullptr;
    ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(lib_handle,
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
    dso_handle_data_map_[lib_handle].kernels_container = kernels_container;

    // register the kernels
    custom_registry.reset();
    custom_registry = std::make_shared<CustomRegistry>();

    for (auto& i : kernels_container->kernels_list) {
      ORT_RETURN_IF_ERROR(custom_registry->RegisterCustomKernel(i));
    }

    // get symbol for GetAllSchemas
    void* get_all_schemas_symbol_handle = nullptr;
    ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(lib_handle,
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
    dso_handle_data_map_[lib_handle].schemas_container = schemas_container;

    // register the schemas if present
    ORT_RETURN_IF_ERROR(custom_registry->RegisterOpSet(schemas_container->schemas_list,
                                                       schemas_container->domain,
                                                       schemas_container->baseline_opset_version,
                                                       schemas_container->opset_version));
    return Status::OK();
  } catch (const std::exception& ex) {
    return Status(ONNXRUNTIME, FAIL, "Caught exception while loading custom ops with message: " + std::string(ex.what()));
  }
}
}  // namespace onnxruntime
