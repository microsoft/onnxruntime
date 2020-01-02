// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "onnxruntime_c_api.h"

#ifdef __cplusplus
#include <WinMLAdapter.h>
using namespace Windows::AI::MachineLearning::Adapter;
#else
struct IWinMLAdapter;
typedef struct IWinMLAdapter IWinMLAdapter;
#endif

ORT_EXPORT STDAPI OrtGetWinMLAdapter(IWinMLAdapter** adapter);

