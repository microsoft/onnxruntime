// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/opwrapper/opwrapper_provider_factory.h"
#include "core/common/gsl_suppress.h"

namespace OrtOpWrapperApis {

ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider, _In_ OrtSessionOptions* session_options,
                    _In_reads_(num_ops) const char* const* op_names,
                    _In_reads_(num_ops) const OrtOpWrapperProviderOptions* const* provider_options,
                    _In_ size_t num_ops);

ORT_API_STATUS_IMPL(CreateProviderOptions, _Outptr_ OrtOpWrapperProviderOptions** out);

ORT_API_STATUS_IMPL(ProviderOptions_Update, _Inout_ OrtOpWrapperProviderOptions* provider_options,
                  _In_reads_(num_options) const char* const* options_keys,
                  _In_reads_(num_options) const char* const* options_values,
                  _In_ size_t num_options);
ORT_API_STATUS_IMPL(ProviderOptions_HasOption, _In_ const OrtOpWrapperProviderOptions* provider_options,
                    _In_ const char* key, _Out_ int* out);
ORT_API_STATUS_IMPL(ProviderOptions_GetOption, _In_ const OrtOpWrapperProviderOptions* provider_options,
                    _In_ const char* key, _Out_ const char** value, _Out_opt_ size_t* length);
ORT_API_STATUS_IMPL(ProviderOptions_Serialize, _In_ const OrtOpWrapperProviderOptions* provider_options,
                    _In_ OrtAllocator* allocator, _Outptr_ const char*** keys, _Outptr_ size_t** key_lengths,
                    _Outptr_ const char*** values, _Outptr_ size_t** value_lengths, _Out_ size_t* num_options);
ORT_API(void, ReleaseOpWrapperProviderOptions, _Frees_ptr_opt_ OrtOpWrapperProviderOptions* provider_options);

ORT_API_STATUS_IMPL(KernelInfo_GetProviderOptions, _In_ const OrtKernelInfo* info, _In_ const char* op_name,
                    _Outptr_ OrtOpWrapperProviderOptions** provider_options);

}  // namespace OrtOpWrapperApis
