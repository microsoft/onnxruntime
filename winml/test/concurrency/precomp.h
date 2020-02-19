//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#include "std.h"
#include "fileHelpers.h"
// Windows pollutes with preprocessor that redefine OPTIONAL.
// Undefine OPTIONAL to get onnx macros to resolve correctly.
#ifdef OPTIONAL
#undef OPTIONAL
#endif
