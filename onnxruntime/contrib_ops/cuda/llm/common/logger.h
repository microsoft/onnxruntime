// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"

#ifdef _MSC_VER
#define PRETTY_FUNCTION __FUNCSIG__
#else
#define PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

#define TRACE_ENTRY_EXIT 1

#if defined(TRACE_ENTRY_EXIT) && TRACE_ENTRY_EXIT
#include <iostream>
#define ORT_LLM_LOG_ENTRY() std::cout << "Entering " << PRETTY_FUNCTION << std::endl;
#define ORT_LLM_LOG_EXIT() std::cout << "Exiting " << PRETTY_FUNCTION << std::endl;
#else
#define ORT_LLM_LOG_ENTRY()
#define ORT_LLM_LOG_EXIT()
#endif

#ifndef NDEBUG
#define ORT_LLM_LOG_TRACE(msg) LOGS_DEFAULT(VERBOSE) << msg
#define ORT_LLM_LOG_DEBUG(msg) LOGS_DEFAULT(VERBOSE) << msg
#else
#define ORT_LLM_LOG_TRACE(msg)
#define ORT_LLM_LOG_DEBUG(msg)
#endif

#define ORT_LLM_LOG_INFO(msg) LOGS_DEFAULT(INFO) << msg
#define ORT_LLM_LOG_WARNING(msg) LOGS_DEFAULT(WARNING) << msg
#define ORT_LLM_LOG_ERROR(msg) LOGS_DEFAULT(ERROR) << msg
