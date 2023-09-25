// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// STL
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING

#include <Windows.h>
#ifdef OPTIONAL
#undef OPTIONAL
#endif

#pragma warning(disable : 4100)

// Needed to work around the fact that OnnxRuntime defines ERROR
#ifdef ERROR
#undef ERROR
#endif
#include "core/session/inference_session.h"
// Restore ERROR define
#define ERROR 0

#ifdef USE_DML
#include <DirectML.h>
#endif USE_DML

#include "core/framework/customregistry.h"
#include "core/framework/allocator_utils.h"
#include "core/session/environment.h"
#include "core/session/IOBinding.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
