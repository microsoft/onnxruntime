// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include <gtest/gtest.h>

#ifdef _WIN32
typedef const wchar_t* PATH_TYPE;
#define TSTR(X) L##X
#else
#define TSTR(X) (X)
typedef const char* PATH_TYPE;
#endif

//empty
static inline void ORT_API_CALL MyLoggingFunction(void*, OrtLoggingLevel, const char*, const char*, const char*, const char*) {
}

