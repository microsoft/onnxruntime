// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif
// Disable C++ linter in this file
// NOLINTBEGIN

struct OrtOpWrapperApi;
typedef struct OrtOpWrapperApi OrtOpWrapperApi;

ORT_RUNTIME_CLASS(OpWrapperProviderOptions);

struct OrtOpWrapperApi {
  /** \brief Append OpWrapper execution provider to the session options.
  *
  * Function allows providing options to multiple custom operator providers.
  * If the OpWrapper execution provider is not enabled, this function returns a failure status.
  *
  * \param[in] session_options  An instance of ::OrtSessionOptions.
  * \param[in] op_names         Array of custom operator names for which to provide options.
  *                             Each name is expected to be a UTF-8 null-terminated string.
  * \param[in] provider_options Array of ::OrtOpWrapperProviderOptions objects. Each element corresponds to the
  *                             op_name at the same index.
  * \param[in] num_ops          Number of custom operators.
  *                             Must be the length of the `op_names` and `provider_options` arrays.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(SessionOptionsAppendExecutionProvider, _In_ OrtSessionOptions* session_options,
                  _In_reads_(num_ops) const char* const* op_names,
                  _In_reads_(num_ops) const OrtOpWrapperProviderOptions* const* provider_options,
                  _In_ size_t num_ops);

  /** \brief Returns a ::OrtOpWrapperProvider object for specifying options for a custom operator provider.
  *
  * \param[out] out Newly created ::OrtOpWrapperProviderOptions object.
  *                 Must be released with OrtOpWrapperApi::ReleaseOpWrapperProviderOptions
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(CreateProviderOptions, _Outptr_ OrtOpWrapperProviderOptions** out);

  /** \brief Update or add options to an ::OrtOpWrapperProviderOptions object.
  *
  * Keys and values are provided in separate parallel arrays of the same size.
  * Overwrites any existing option values.
  *
  * \param[in] provider_options The instance of ::OrtOpWrapperProviderOptions to update.
  * \param[in] options_keys     Array of UTF-8 null-terminated strings representing provider options keys.
  * \param[in] options_values   Array of UTF-8 null-terminated strings representing provider options values.
  * \param[in] num_options      Number of elements in the `options_keys` and `options_values` arrays.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(ProviderOptions_Update, _Inout_ OrtOpWrapperProviderOptions* provider_options,
                  _In_reads_(num_options) const char* const* options_keys,
                  _In_reads_(num_options) const char* const* options_values,
                  _In_ size_t num_options);

  /** \brief Checks if the specified option exists in the ::OrtOpWrapperProviderOptions.
  *
  * \param[in]  provider_options An instance of ::OrtOpWrapperProviderOptions.
  * \param[in]  key              The option whose existence to check.
  * \param[out] out              Set to 1 if the option exists, and 0 otherwise.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(ProviderOptions_HasOption, _In_ const OrtOpWrapperProviderOptions* provider_options,
                  _In_ const char* key, _Out_ int* out);

  /** \brief Get the value of a provider option in ::OrtOpWrapperProviderOptions.
  *
  * Returns a failure status if the option denoted by `key` does not exist.
  * Use OrtOpWrapperApi::ProviderOptions_HasOption to check if the option exists.
  *
  * \param[in]  provider_options An instance of ::OrtOpWrapperProviderOptions.
  * \param[in]  key              The option whose value to get.
  * \param[out] value            Pointer set to the non-modifiable UTF-8 null-terminated string representing the
  *                              option's value. Do not free this value.
  *                              Invalidated if owning ::OrtOpWrapperProvider is destroyed or the option is updated.
  * \param[out] length           Optional pointer set to the value string's length (not counting null-terminator).
  *                              Ignored if NULL.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(ProviderOptions_GetOption, _In_ const OrtOpWrapperProviderOptions* provider_options,
                  _In_ const char* key, _Out_ const char** value, _Out_opt_ size_t* length);

  /** \brief Returns serialized provider options keys and values.
  *
  * The key and value strings returned by this function are owned by the ::OrtOpWrapperProviderOptions object
  * and should not be modified. The keys and values are invalidated when the ::OrtOpWrapperProviderOptions is
  * destroyed or updated.
  *
  * \param[in] provider_options An instance of ::OrtOpWrapperProviderOptions.
  * \param[in] allocator        An instance of ::OrtAllocator used to allocate the `keys`, `key_lengths`, `values`,
  *                             and `value_lengths` arrays.
  * \param[out] keys            Pointer set to an array of non-modifiable UTF-8 null-terminated strings representing
  *                             the option keys. The array must be freed with the provided allocator.
  * \param[out] key_lengths     Pointer set to an array of key string lengths.
  *                             The array must be freed with the provided allocator.
  * \param[out] values          Pointer set to an array of non-modifiable UTF-8 null-terminated strings representing
  *                             the option keys. The array must be freed with the provided allocator.
  * \param[out] value_lengths   Pointer set to an array of key string lengths.
  *                             The array must be freed with the provided allocator.
  * \param[out] num_options     Set to the number of provider options, which is equal to the length of all
  *                             output arrays.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(ProviderOptions_Serialize, _In_ const OrtOpWrapperProviderOptions* provider_options,
                  _In_ OrtAllocator* allocator, _Outptr_ const char*** keys, _Outptr_ size_t** key_lengths,
                  _Outptr_ const char*** values, _Outptr_ size_t** value_lengths, _Out_ size_t* num_options);

  /* \brief: Release provider options
  *
  * \param[in] OrtOpWrapperProviderOptions
  *
  * \since Version 1.13
  */
  ORT_CLASS_RELEASE(OpWrapperProviderOptions);

  /** \brief Get an instance of ::OrtOpWrapperProviderOptions from an instance of ::OrtKernelInfo.
  *
  * \param[in]  info             An instance of ::OrtKernelInfo.
  * \param[in]  op_name          The name of the custom operator provider as a UTF-8 null-terminated string.
  * \param[out] provider_options A copy of the provider options.
  *                              Must free with OrtOpWrapperApi::ReleaseOpWrapperProviderOptions.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(KernelInfo_GetProviderOptions, _In_ const OrtKernelInfo* info, _In_ const char* op_name,
                  _Outptr_ OrtOpWrapperProviderOptions** provider_options);

#ifdef __cplusplus
  OrtOpWrapperApi(const OrtOpWrapperApi&) = delete;  // Prevent users from accidentally copying the API structure,
                                                     // it should always be passed as a pointer
#endif
};

#ifdef __cplusplus
}
#endif
// NOLINTEND
