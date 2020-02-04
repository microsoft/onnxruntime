// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

struct WinmlAdapterApi;
typedef struct WinmlAdapterApi WinmlAdapterApi;

ORT_EXPORT const WinmlAdapterApi* ORT_API_CALL OrtGetWinMLAdapter(_In_ const OrtApi* ort_api) NO_EXCEPTION;