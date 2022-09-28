// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

struct OrtOpWrapperApi;
typedef struct OrtOpWrapperApi OrtOpWrapperApi;

ORT_RUNTIME_CLASS(OpWrapperProviderOptions);

struct OrtOpWrapperApi {
  /** \brief Append OpWrapper execution provider to the session options
  *
  * If OpWrapper is not enabled (due to build options), this function returns a failure status.
  *
  * \param[in] session_options The session options.
  * \param[in] op_names Array of custom operator names for which to provide options.
  * \param[in] provider_options Array of opaque provider options objects. Each element corresponds to an op_name.
  * \param[in] num_ops Number of custom operators. Must be the length of the `op_names` and `provider_options` arrays.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_OpWrapper, _In_ OrtSessionOptions* session_options,
                  _In_reads_(num_ops) const char* const* op_names,
                  _In_reads_(num_ops) const OrtOpWrapperProviderOptions* const* provider_options,
                  _In_ size_t num_ops);

  /** \brief Create an opaque provider options object for a custom operator provider.
  *
  * \param[out] out Newly created ::OrtOpWrapperProviderOptions object.
                    Must be released with OrtOpWrapperApi::ReleaseOpWrapperProviderOptions
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(CreateProviderOptions, _Outptr_ OrtOpWrapperProviderOptions** out);

  /** \brief Update options for a custom operator provider.
  *
  * Keys and values are null-terminated strings. Overwrites existing options.
  *
  * \param[in] provider_options The options object to update.
  * \param[in] options_keys Array of UTF-8 null-terminated strings for provider options keys
  * \param[in] options_values Array of UTF-8 null-terminated strings for provider options values
  * \param[in] num_options Number of elements in the `options_keys` and `options_values` arrays.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(ProviderOptions_Update, _Inout_ OrtOpWrapperProviderOptions* provider_options,
                  _In_reads_(num_options) const char* const* options_keys,
                  _In_reads_(num_options) const char* const* options_values,
                  _In_ size_t num_options);

  /** \brief Check if a provider option exists and get its size.
  *
  * If the option exists, the option value's size (string length + 1) is returned
  * in the `value_size` parameter.
  *
  * If the option does not exist, the `value_size` parameter is set to 0.
  *
  * \param[in] provider_options
  * \param[in] key The option to check.
  * \param[out] value_size The option value's size if it exists. See above comments for details.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(ProviderOptions_HasOption, _In_ const OrtOpWrapperProviderOptions* provider_options,
                  _In_ const char* key, _Inout_ size_t* value_size);

  /** \brief Get a the value of a provider option.
  *
  * If `value` is a nullptr, `value_size` is set to the length of the option value (plus 1),
  * and a success status is returned.
  *
  * If the `value_size` parameter is large enough, `value_size` is set to the length of the option value (plus 1),
  * the provided buffer is filled with the null-terminated option value, and a success status is returned.
  *
  * If `value_size` is less than the required size and `value` is not a nullptr,
  * `value_size` is set to the required size, and a failure status is returned.
  *
  * \param[in] provider_options
  * \param[in] key The option to get.
  * \param[out] value The buffer into which to store the null-terminated option value.
  * \param[out] value_size The value buffer's size. See above comments for details.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(ProviderOptions_GetOption, _In_ const OrtOpWrapperProviderOptions* provider_options,
                  _In_ const char* key, _Out_opt_ char* value, _Inout_ size_t* value_size);

  /** \brief Get serialized provider options keys and values.
  *
  * If either `keys` or `values` are nullptr, the values of `keys_size` and `values_size`
  * are set to the minimum buffer sizes necessary to store all serialized keys and values respectively,
  * and a success status is returned.
  *
  * If the `keys_size` and `values_size` parameters are both large enough,
  * the values of `keys_size` and `values_size` are set to the actual serialized sizes, the provided buffers
  * are filled with the corresponding keys or values (separated by '\0'), and a success status is returned.
  *
  * If either of the `keys_size` or `values_size` parameters is less than the required size and
  * both `keys` and `values` are not nullptr, the values `keys_size` and `values_size` are set to the required
  * sizes and a failure status is returned.
  *
  * \param[in] provider_options
  * \param[out] keys The buffer into which to serialize keys (separated by '\0').
  * \param[out] keys_size See above comments for details.
  * \param[out] values The buffer into which to serialize values (separated by '\0').
  * \param[out] values_size See above comments for details.
  * \param[out] num_options Set to the number of provider options if not a nullptr.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  * \since Version 1.13
  */
  ORT_API2_STATUS(ProviderOptions_Serialize, _In_ const OrtOpWrapperProviderOptions* provider_options,
                  _Out_opt_ char* keys, _Inout_ size_t* keys_size,
                  _Out_opt_ char* values, _Inout_ size_t* values_size, _Out_opt_ size_t* num_options);

  /* \brief: Release provider options
  *
  * \param[in] OrtOpWrapperProviderOptions
  *
  * \since Version 1.13
  */
  ORT_CLASS_RELEASE(OpWrapperProviderOptions);

  /** \brief Get OpWrapper execution provider options from an instance of OrtKernelInfo.
  *
  * \param[in] info An instance of OrtKernelInfo.
  * \param[in] op_name The name of the custom operator provider for which to get options.
  * \param[out] provider_options A copy of the provider options. Must free with OrtOpWrapperApi::ReleaseOpWrapperProviderOptions.
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
