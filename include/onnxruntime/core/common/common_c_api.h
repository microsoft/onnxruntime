// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
// This header file defines common used macros/typedef/... for c API.
// NOTE: Don't include this file directly, include common.h, unless you are using c API only.

#ifdef _WIN32
#define ORT_MUST_USE_RESULT
#else
#define ORT_MUST_USE_RESULT __attribute__((warn_unused_result))
#endif
