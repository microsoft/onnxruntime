// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef __cplusplus
#include <WinMLAdapter.h>
using namespace Windows::AI::MachineLearning::Adapter;

#else
struct IWinMLAdapter;
typedef struct IWinMLAdapter IWinMLAdapter;
#endif

#ifdef __cplusplus
extern "C" {
#endif

HRESULT STDMETHODCALLTYPE OrtGetWinMLAdapter(IWinMLAdapter** adapter);

#ifdef __cplusplus
}
#endif
