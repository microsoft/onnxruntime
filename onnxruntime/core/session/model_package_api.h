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

ORT_API_STATUS_IMPL(ModelPackage_GetComponentCount,
                    _In_ const OrtModelPackageContext* ctx,
                    _Out_ size_t* out_count);

ORT_API_STATUS_IMPL(ModelPackage_GetComponentNames,
                    _In_ const OrtModelPackageContext* ctx,
                    _Outptr_result_buffer_maybenull_(*out_count) const char* const** out_names,
                    _Out_ size_t* out_count);

ORT_API_STATUS_IMPL(ModelPackage_GetVariantCount,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _Out_ size_t* out_count);

ORT_API_STATUS_IMPL(ModelPackage_GetVariantNames,
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

ORT_API_STATUS_IMPL(CreateSession,
                    _In_ const OrtEnv* env,
                    _In_ OrtModelPackageComponentContext* ctx,
                    _In_opt_ const OrtSessionOptions* session_options,
                    _Outptr_ OrtSession** session);

ORT_API_STATUS_IMPL(ModelPackage_GetVariantEpName,
                    _In_ const OrtModelPackageContext* ctx,
                    _In_ const char* component_name,
                    _In_ const char* variant_name,
                    _Outptr_result_maybenull_ const char** out_ep);

ORT_API_STATUS_IMPL(ModelPackage_GetSchemaVersion,
                    _In_ const OrtModelPackageContext* ctx,
                    _Out_ int64_t* out_version);

ORT_API_STATUS_IMPL(ModelPackageComponent_GetSelectedVariantName,
                    _In_ const OrtModelPackageComponentContext* ctx,
                    _Outptr_ const char** out_name);

}  // namespace OrtModelPackageAPI
