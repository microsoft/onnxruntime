// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#endif
#include "std.h"

// Windows pollutes with preprocessor that redefine OPTIONAL.
// Undefine OPTIONAL to get onnx macros to resolve correctly.
#ifdef OPTIONAL
#undef OPTIONAL
#endif

#include <wrl/client.h>
#include <wrl/implements.h>

#include "fileHelpers.h"
#include "dllload.h"
#include <thread>
