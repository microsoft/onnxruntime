// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifndef NDEBUG
#ifdef ONNXRUNTIME_ENABLE_MEMLEAK_CHECK
void* DebugHeapAlloc(size_t size, unsigned framesToSkip = 0);
void* DebugHeapReAlloc(void* p, size_t size);
void DebugHeapFree(void* p) noexcept;

#define calloc CallocNotImplemented
#define malloc DebugHeapAlloc
#define realloc DebugHeapReAlloc
#define free DebugHeapFree
#endif
#endif
