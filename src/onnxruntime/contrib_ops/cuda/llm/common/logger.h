// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"

#ifdef _MSC_VER
#define PRETTY_FUNCTION __FUNCSIG__
#else
#define PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

#define ORT_LLM_VERBOSE 0  // Set to 1 for verbose, 2 for max verbosity

#if ORT_LLM_VERBOSE
#include <iostream>
#define ORT_LLM_LOG_ENTRY() std::cout << "Enter " << PRETTY_FUNCTION << std::endl;
#else
#define ORT_LLM_LOG_ENTRY()
#endif

#if ORT_LLM_VERBOSE
#define ORT_LLM_LOG_TRACE(msg) LOGS_DEFAULT(VERBOSE) << msg
#define ORT_LLM_LOG_DEBUG(msg) LOGS_DEFAULT(VERBOSE) << msg
#else
#define ORT_LLM_LOG_TRACE(msg)
#define ORT_LLM_LOG_DEBUG(msg)
#endif

#define ORT_LLM_LOG_INFO(msg) LOGS_DEFAULT(INFO) << msg
#define ORT_LLM_LOG_WARNING(msg) LOGS_DEFAULT(WARNING) << msg
#define ORT_LLM_LOG_ERROR(msg) LOGS_DEFAULT(ERROR) << msg
