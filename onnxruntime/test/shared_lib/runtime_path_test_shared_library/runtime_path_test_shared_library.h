// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// To make symbols visible on macOS/iOS
#if defined(__APPLE__)
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

#if defined(_WIN32)
#define PATH_CHAR_T wchar_t
#else
#define PATH_CHAR_T char
#endif

extern "C" {
//
// Public symbols
//

// Gets the runtime path of the shared library - i.e., the shared library file's parent directory.
EXPORT_SYMBOL const PATH_CHAR_T* OrtTestGetSharedLibraryRuntimePath(void);
}
