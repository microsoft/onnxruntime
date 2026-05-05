// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

namespace OrtModelPackageAPI {

ORT_API(const OrtModelPackageApi*, GetModelPackageApi);

ORT_API(void, ReleaseModelPackageOptions, _Frees_ptr_opt_ OrtModelPackageOptions*);
ORT_API_STATUS_IMPL(CreateModelPackageOptionsFromSessionOptions,
                    _In_ const OrtEnv* env,
                    _In_ const OrtSessionOptions* session_options,
                    _Outptr_ OrtModelPackageOptions** out);

ORT_API(void, ReleaseModelPackageContext, _Frees_ptr_opt_ OrtModelPackageContext*);
ORT_API_STATUS_IMPL(CreateModelPackageContext,
                    _In_ const ORTCHAR_T* package_root,
                    _Outptr_ OrtModelPackageContext** out);

ORT_API_STATUS_IMPL(ModelPackage_GetComponentModelCount,
                    _In_ const OrtModelPackageContext* ctx,
                    _Out_ size_t* out_count);

ORT_API_STATUS_IMPL(ModelPackage_GetComponentModelNames,
                    _In_ const OrtModelPackageContext* ctx,
                    _Outptr_result_buffer_maybenull_(*out_count) const char* const** out_names,
                    _Out_ size_t* out_count);

ORT_API_STATUS_IMPL(ModelPackage_GetModelVariantCount,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _Out_ size_t* out_count);

ORT_API_STATUS_IMPL(ModelPackage_GetModelVariantNames,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _Outptr_result_buffer_maybenull_(*out_count) const char* const** out_variant_names,
                    _Out_ size_t* out_count);

ORT_API_STATUS_IMPL(SelectComponent,
                    _In_ const OrtModelPackageContext* context,
                    _In_ const char* component_name,
                    _In_ const OrtModelPackageOptions* options,
                    _Outptr_ OrtModelPackageComponentContext** out);

ORT_API(void, ReleaseModelPackageComponentContext,
        _Frees_ptr_opt_ OrtModelPackageComponentContext* ctx);

ORT_API_STATUS_IMPL(ModelPackageComponent_GetSelectedVariantFolderPath,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _Outptr_ const ORTCHAR_T** folder_path);

ORT_API_STATUS_IMPL(ModelPackageComponent_GetSelectedVariantFileCount,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _Out_ size_t* num_files);

ORT_API_STATUS_IMPL(ModelPackageComponent_GetSelectedVariantFilePath,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _In_ size_t file_idx,
                    _Outptr_ const ORTCHAR_T** out_path);

ORT_API_STATUS_IMPL(ModelPackageComponent_GetSelectedVariantFileSessionOptions,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _In_ size_t file_idx,
                    _Outptr_result_buffer_maybenull_(*num_entries) const char* const** option_keys,
                    _Outptr_result_buffer_maybenull_(*num_entries) const char* const** option_values,
                    _Out_ size_t* num_entries);

ORT_API_STATUS_IMPL(ModelPackageComponent_GetSelectedVariantFileProviderOptions,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _In_ size_t file_idx,
                    _Outptr_result_buffer_maybenull_(*num_entries) const char* const** option_keys,
                    _Outptr_result_buffer_maybenull_(*num_entries) const char* const** option_values,
                    _Out_ size_t* num_entries);

ORT_API_STATUS_IMPL(CreateSession,
                    _In_ const OrtEnv* env,
                    _In_ OrtModelPackageComponentContext* ctx,
                    _In_opt_ const OrtSessionOptions* session_options,
                    _Outptr_ OrtSession** session);

}  // namespace OrtModelPackageAPI