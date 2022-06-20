// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param[in] dnnl_options configuration parameters for oneDnnl EP creation.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Dnnl,
               _In_ OrtSessionOptions* options, _In_ const OrtDnnlProviderOptions* dnnl_options);

#ifdef __cplusplus
}
#endif
