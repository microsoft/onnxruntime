// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(_MSC_VER)
#define ORT_FORCEINLINE __forceinline
#else
#define ORT_FORCEINLINE __attribute__((always_inline)) inline
#endif
