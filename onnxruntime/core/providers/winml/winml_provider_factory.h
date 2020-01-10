// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "onnxruntime_c_api.h"
//
#ifdef __cplusplus
#include <WinMLAdapter.h>
using namespace Windows::AI::MachineLearning::Adapter;
#else
struct IWinMLAdapter;
typedef struct IWinMLAdapter IWinMLAdapter;
#endif

ORT_EXPORT STDAPI OrtGetWinMLAdapter(IWinMLAdapter** adapter);

// Add at the end when we remove the winmladapter.h
//struct WinmlAdapterApi;
//typedef struct WinmlAdapterApi WinmlAdapterApi;
//
//ORT_EXPORT const WinmlAdapterApi* ORT_API_CALL GetWinmlAdapterApi(_In_ const OrtApi* ort_api)NO_EXCEPTION;