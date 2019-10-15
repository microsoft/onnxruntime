// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_config.h"
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#ifdef HAS_IGNORED_ATTRIBUTES
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#ifdef HAS_DEPRECATED_COPY
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#endif
#else
#pragma warning(push)
#pragma warning(disable : 4100)  //'identifier' : unreferenced formal parameter
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

#include "public/gemmlowp.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif