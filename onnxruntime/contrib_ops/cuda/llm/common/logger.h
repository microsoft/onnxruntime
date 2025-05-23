// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"

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
