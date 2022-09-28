// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/opwrapper/opwrapper_provider_factory.h"
#include "core/common/gsl_suppress.h"

namespace OrtOpWrapperApis {

ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_OpWrapper, _In_ OrtSessionOptions* session_options,
                    _In_reads_(num_ops) const char* const* op_names,
                    _In_reads_(num_ops) const OrtOpWrapperProviderOptions* const* provider_options,
                    _In_ size_t num_ops);

ORT_API_STATUS_IMPL(CreateProviderOptions, _Outptr_ OrtOpWrapperProviderOptions** out);

ORT_API_STATUS_IMPL(ProviderOptions_Update, _Inout_ OrtOpWrapperProviderOptions* provider_options,
                  _In_reads_(num_options) const char* const* options_keys,
                  _In_reads_(num_options) const char* const* options_values,
                  _In_ size_t num_options);
ORT_API_STATUS_IMPL(ProviderOptions_HasOption, _In_ const OrtOpWrapperProviderOptions* provider_options,
                    _In_ const char* key, _Inout_ size_t* value_size);
ORT_API_STATUS_IMPL(ProviderOptions_GetOption, _In_ const OrtOpWrapperProviderOptions* provider_options,
                    _In_ const char* key, _Out_opt_ char* value, _Inout_ size_t* value_size);
ORT_API_STATUS_IMPL(ProviderOptions_Serialize, _In_ const OrtOpWrapperProviderOptions* provider_options,
                    _Out_opt_ char* keys, _Inout_ size_t* keys_size,
                    _Out_opt_ char* values, _Inout_ size_t* values_size, _Out_opt_ size_t* num_options);
ORT_API(void, ReleaseOpWrapperProviderOptions, _Frees_ptr_opt_ OrtOpWrapperProviderOptions* provider_options);

ORT_API_STATUS_IMPL(KernelInfo_GetProviderOptions, _In_ const OrtKernelInfo* info, _In_ const char* op_name,
                    _Outptr_ OrtOpWrapperProviderOptions** provider_options);

}  // namespace OrtOpWrapperApis
