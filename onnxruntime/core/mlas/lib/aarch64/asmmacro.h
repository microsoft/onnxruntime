/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    asmmacro.h

Abstract:

    This module implements common macros for the assembly modules.

--*/

#if defined(__APPLE__)
#define C_UNDERSCORE(symbol) _##symbol
#else
#define C_UNDERSCORE(symbol) symbol
#endif
