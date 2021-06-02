// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

// Avoid  linking to pythonX_d.lib on Windows in debug build
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable: 4510 4610 4512 4005)
#ifdef _DEBUG
#define ORT_DISABLE_INCLUDE_DEBUG_PYTHON_LIB
#undef _DEBUG
#endif
#endif

#include <Python.h>

#ifdef _WIN32
#ifdef ORT_DISABLE_INCLUDE_DEBUG_PYTHON_LIB
#define _DEBUG
#undef ORT_DISABLE_INCLUDE_DEBUG_PYTHON_LIB
#endif
#pragma warning(pop)
#endif