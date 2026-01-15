// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/interop_api.h"

#if !defined(ORT_MINIMAL_BUILD)
#include <memory>

#include "core/session/ort_apis.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_env.h"
#include "core/session/abi_devices.h"
#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#else
#include "core/framework/error_code_helper.h"
#include "core/session/ort_apis.h"
#endif  // !defined(ORT_MINIMAL_BUILD)

using namespace onnxruntime;

#if !defined(ORT_MINIMAL_BUILD)

// Wrapper class for OrtExternalResourceImporterImpl
namespace {
struct ExternalResourceImporterWrapper {
  const OrtEpDevice* ep_device;
  OrtExternalResourceImporterImpl* impl;

  ExternalResourceImporterWrapper(const OrtEpDevice* device, OrtExternalResourceImporterImpl* importer)
      : ep_device(device), impl(importer) {}

  ~ExternalResourceImporterWrapper() {
    if (impl && impl->Release) {
      impl->Release(impl);
    }
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExternalResourceImporterWrapper);
};

}  // namespace

ORT_API_STATUS_IMPL(OrtInteropAPI::CreateExternalResourceImporterForDevice, _In_ const OrtEpDevice* ep_device,
                    _Outptr_result_maybenull_ OrtExternalResourceImporter** out_importer) {
  API_IMPL_BEGIN
  if (ep_device == nullptr || out_importer == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "ep_device and out_importer must be provided.");
  }

  *out_importer = nullptr;

  // OrtEpFactory::CreateExternalResourceImporterForDevice was added in ORT 1.24.
  const auto* factory = ep_device->ep_factory;
  if (factory == nullptr ||
      factory->ort_version_supported < 24 ||
      factory->CreateExternalResourceImporterForDevice == nullptr) {
    // EP doesn't support this optional feature
    return nullptr;
  }

  OrtExternalResourceImporterImpl* impl = nullptr;
  ORT_API_RETURN_IF_ERROR(factory->CreateExternalResourceImporterForDevice(
      ep_device->GetMutableFactory(),
      ep_device,
      &impl));

  if (impl == nullptr) {
    // EP doesn't support this for the specific device
    return nullptr;
  }

  auto wrapper = std::make_unique<ExternalResourceImporterWrapper>(ep_device, impl);
  *out_importer = reinterpret_cast<OrtExternalResourceImporter*>(wrapper.release());

  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtInteropAPI::ReleaseExternalResourceImporter, _Frees_ptr_opt_ OrtExternalResourceImporter* importer) {
#if !defined(ORT_MINIMAL_BUILD)
  delete reinterpret_cast<ExternalResourceImporterWrapper*>(importer);
#else
  ORT_UNUSED_PARAMETER(importer);
#endif  // !defined(ORT_MINIMAL_BUILD)
}

ORT_API_STATUS_IMPL(OrtInteropAPI::CanImportMemory, _In_ const OrtExternalResourceImporter* importer,
                    _In_ OrtExternalMemoryHandleType handle_type,
                    _Out_ bool* out_supported) {
  API_IMPL_BEGIN
  if (importer == nullptr || out_supported == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "importer and out_supported must be provided.");
  }

  auto* wrapper = reinterpret_cast<const ExternalResourceImporterWrapper*>(importer);
  if (wrapper->impl == nullptr || wrapper->impl->CanImportMemory == nullptr) {
    *out_supported = false;
    return nullptr;
  }

  *out_supported = wrapper->impl->CanImportMemory(wrapper->impl, handle_type);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtInteropAPI::ImportMemory, _In_ OrtExternalResourceImporter* importer,
                    _In_ const OrtExternalMemoryDescriptor* desc,
                    _Outptr_ OrtExternalMemoryHandle** out_handle) {
  API_IMPL_BEGIN
  if (importer == nullptr || desc == nullptr || out_handle == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "importer, desc, and out_handle must be provided.");
  }

  auto* wrapper = reinterpret_cast<ExternalResourceImporterWrapper*>(importer);
  if (wrapper->impl == nullptr || wrapper->impl->ImportMemory == nullptr) {
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "External memory import is not supported by this EP.");
  }

  // EP creates derived type and returns base pointer. EP owns the handle lifetime.
  OrtExternalMemoryHandle* handle = nullptr;
  ORT_API_RETURN_IF_ERROR(wrapper->impl->ImportMemory(wrapper->impl, desc, &handle));

  if (handle == nullptr) {
    return OrtApis::CreateStatus(ORT_FAIL, "ImportMemory returned null handle.");
  }

  *out_handle = handle;

  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtInteropAPI::ReleaseExternalMemoryHandle, _Frees_ptr_opt_ OrtExternalMemoryHandle* handle) {
#if !defined(ORT_MINIMAL_BUILD)
  if (handle != nullptr && handle->Release != nullptr) {
    handle->Release(handle);
  }
#else
  ORT_UNUSED_PARAMETER(handle);
#endif  // !defined(ORT_MINIMAL_BUILD)
}

ORT_API_STATUS_IMPL(OrtInteropAPI::CreateTensorFromMemory, _In_ OrtExternalResourceImporter* importer,
                    _In_ const OrtExternalMemoryHandle* mem_handle,
                    _In_ const OrtExternalTensorDescriptor* tensor_desc,
                    _Outptr_ OrtValue** out_tensor) {
  API_IMPL_BEGIN
  if (importer == nullptr || mem_handle == nullptr || tensor_desc == nullptr || out_tensor == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "importer, mem_handle, tensor_desc, and out_tensor must be provided.");
  }

  auto* imp_wrapper = reinterpret_cast<ExternalResourceImporterWrapper*>(importer);

  if (imp_wrapper->impl == nullptr || imp_wrapper->impl->CreateTensorFromMemory == nullptr) {
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "CreateTensorFromMemory is not supported by this EP.");
  }

  OrtValue* tensor = nullptr;
  ORT_API_RETURN_IF_ERROR(imp_wrapper->impl->CreateTensorFromMemory(imp_wrapper->impl, mem_handle, tensor_desc, &tensor));

  *out_tensor = tensor;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtInteropAPI::CanImportSemaphore, _In_ const OrtExternalResourceImporter* importer,
                    _In_ OrtExternalSemaphoreType type,
                    _Out_ bool* out_supported) {
  API_IMPL_BEGIN
  if (importer == nullptr || out_supported == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "importer and out_supported must be provided.");
  }

  auto* wrapper = reinterpret_cast<const ExternalResourceImporterWrapper*>(importer);
  if (wrapper->impl == nullptr || wrapper->impl->CanImportSemaphore == nullptr) {
    *out_supported = false;
    return nullptr;
  }

  *out_supported = wrapper->impl->CanImportSemaphore(wrapper->impl, type);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtInteropAPI::ImportSemaphore, _In_ OrtExternalResourceImporter* importer,
                    _In_ const OrtExternalSemaphoreDescriptor* desc,
                    _Outptr_ OrtExternalSemaphoreHandle** out_handle) {
  API_IMPL_BEGIN
  if (importer == nullptr || desc == nullptr || out_handle == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "importer, desc, and out_handle must be provided.");
  }

  auto* wrapper = reinterpret_cast<ExternalResourceImporterWrapper*>(importer);
  if (wrapper->impl == nullptr || wrapper->impl->ImportSemaphore == nullptr) {
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "External semaphore import is not supported by this EP.");
  }

  // EP creates derived type and returns base pointer. EP owns the handle lifetime.
  OrtExternalSemaphoreHandle* handle = nullptr;
  ORT_API_RETURN_IF_ERROR(wrapper->impl->ImportSemaphore(wrapper->impl, desc, &handle));

  if (handle == nullptr) {
    return OrtApis::CreateStatus(ORT_FAIL, "ImportSemaphore returned null handle.");
  }

  *out_handle = handle;

  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtInteropAPI::ReleaseExternalSemaphoreHandle, _Frees_ptr_opt_ OrtExternalSemaphoreHandle* handle) {
#if !defined(ORT_MINIMAL_BUILD)
  if (handle != nullptr && handle->Release != nullptr) {
    handle->Release(handle);
  }
#else
  ORT_UNUSED_PARAMETER(handle);
#endif  // !defined(ORT_MINIMAL_BUILD)
}

ORT_API_STATUS_IMPL(OrtInteropAPI::WaitSemaphore, _In_ OrtExternalResourceImporter* importer,
                    _In_ OrtExternalSemaphoreHandle* semaphore_handle,
                    _In_ OrtSyncStream* stream,
                    _In_ uint64_t value) {
  API_IMPL_BEGIN
  if (importer == nullptr || semaphore_handle == nullptr || stream == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "importer, semaphore_handle, and stream must be provided.");
  }

  auto* imp_wrapper = reinterpret_cast<ExternalResourceImporterWrapper*>(importer);

  if (imp_wrapper->impl == nullptr || imp_wrapper->impl->WaitSemaphore == nullptr) {
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "WaitSemaphore is not supported by this EP.");
  }

  ORT_API_RETURN_IF_ERROR(imp_wrapper->impl->WaitSemaphore(imp_wrapper->impl, semaphore_handle, stream, value));

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtInteropAPI::SignalSemaphore, _In_ OrtExternalResourceImporter* importer,
                    _In_ OrtExternalSemaphoreHandle* semaphore_handle,
                    _In_ OrtSyncStream* stream,
                    _In_ uint64_t value) {
  API_IMPL_BEGIN
  if (importer == nullptr || semaphore_handle == nullptr || stream == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "importer, semaphore_handle, and stream must be provided.");
  }

  auto* imp_wrapper = reinterpret_cast<ExternalResourceImporterWrapper*>(importer);

  if (imp_wrapper->impl == nullptr || imp_wrapper->impl->SignalSemaphore == nullptr) {
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "SignalSemaphore is not supported by this EP.");
  }

  ORT_API_RETURN_IF_ERROR(imp_wrapper->impl->SignalSemaphore(imp_wrapper->impl, semaphore_handle, stream, value));

  return nullptr;
  API_IMPL_END
}

#else  // defined(ORT_MINIMAL_BUILD)

ORT_API_STATUS_IMPL(OrtInteropAPI::CreateExternalResourceImporterForDevice, _In_ const OrtEpDevice* ep_device,
                    _Outptr_result_maybenull_ OrtExternalResourceImporter** out_importer) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(ep_device);
  ORT_UNUSED_PARAMETER(out_importer);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Interop API is not supported in this build");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtInteropAPI::CanImportMemory, _In_ const OrtExternalResourceImporter* importer,
                    _In_ OrtExternalMemoryHandleType handle_type,
                    _Out_ bool* out_supported) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(importer);
  ORT_UNUSED_PARAMETER(handle_type);
  ORT_UNUSED_PARAMETER(out_supported);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Interop API is not supported in this build");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtInteropAPI::ImportMemory, _In_ OrtExternalResourceImporter* importer,
                    _In_ const OrtExternalMemoryDescriptor* desc,
                    _Outptr_ OrtExternalMemoryHandle** out_handle) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(importer);
  ORT_UNUSED_PARAMETER(desc);
  ORT_UNUSED_PARAMETER(out_handle);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Interop API is not supported in this build");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtInteropAPI::CreateTensorFromMemory, _In_ OrtExternalResourceImporter* importer,
                    _In_ const OrtExternalMemoryHandle* mem_handle,
                    _In_ const OrtExternalTensorDescriptor* tensor_desc,
                    _Outptr_ OrtValue** out_tensor) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(importer);
  ORT_UNUSED_PARAMETER(mem_handle);
  ORT_UNUSED_PARAMETER(tensor_desc);
  ORT_UNUSED_PARAMETER(out_tensor);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Interop API is not supported in this build");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtInteropAPI::CanImportSemaphore, _In_ const OrtExternalResourceImporter* importer,
                    _In_ OrtExternalSemaphoreType type,
                    _Out_ bool* out_supported) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(importer);
  ORT_UNUSED_PARAMETER(type);
  ORT_UNUSED_PARAMETER(out_supported);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Interop API is not supported in this build");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtInteropAPI::ImportSemaphore, _In_ OrtExternalResourceImporter* importer,
                    _In_ const OrtExternalSemaphoreDescriptor* desc,
                    _Outptr_ OrtExternalSemaphoreHandle** out_handle) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(importer);
  ORT_UNUSED_PARAMETER(desc);
  ORT_UNUSED_PARAMETER(out_handle);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Interop API is not supported in this build");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtInteropAPI::WaitSemaphore, _In_ OrtExternalResourceImporter* importer,
                    _In_ OrtExternalSemaphoreHandle* semaphore_handle,
                    _In_ OrtSyncStream* stream,
                    _In_ uint64_t value) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(importer);
  ORT_UNUSED_PARAMETER(semaphore_handle);
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(value);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Interop API is not supported in this build");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtInteropAPI::SignalSemaphore, _In_ OrtExternalResourceImporter* importer,
                    _In_ OrtExternalSemaphoreHandle* semaphore_handle,
                    _In_ OrtSyncStream* stream,
                    _In_ uint64_t value) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(importer);
  ORT_UNUSED_PARAMETER(semaphore_handle);
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(value);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Interop API is not supported in this build");
  API_IMPL_END
}

#endif  // !defined(ORT_MINIMAL_BUILD)

static constexpr OrtInteropApi ort_interop_api = {
    // NOTE: Application compatibility with newer versions of ORT depends on the Api order within this struct so
    // all new functions must be added at the end, and no functions that already exist in an officially released version
    // of ORT can be reordered or removed.

    &OrtInteropAPI::CreateExternalResourceImporterForDevice,
    &OrtInteropAPI::ReleaseExternalResourceImporter,
    &OrtInteropAPI::CanImportMemory,
    &OrtInteropAPI::ImportMemory,
    &OrtInteropAPI::ReleaseExternalMemoryHandle,
    &OrtInteropAPI::CreateTensorFromMemory,
    &OrtInteropAPI::CanImportSemaphore,
    &OrtInteropAPI::ImportSemaphore,
    &OrtInteropAPI::ReleaseExternalSemaphoreHandle,
    &OrtInteropAPI::WaitSemaphore,
    &OrtInteropAPI::SignalSemaphore,
    // End of Version 24 - DO NOT MODIFY ABOVE
};

// Checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtInteropApi, SignalSemaphore) / sizeof(void*) == 10,
              "Size of version 24 Api cannot change");  // initial version in ORT 1.24

ORT_API(const OrtInteropApi*, OrtInteropAPI::GetInteropApi) {
  return &ort_interop_api;
}
