// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// need to include corecrt.h before pybind so that the _STL_ASSERT macros are defined.
// pybind11/pybind11.h includes pybind11/detail/common.h, which undefines _DEBUG
// whilst including the Python headers, which alters how the _STL_ASSERT macros are created.
//
// see https://github.com/microsoft/onnxruntime/issues/9735
#if defined(_MSC_VER) && defined(_DEBUG) && _MSC_VER >= 1930
#include <corecrt.h>
#endif

#include <pybind11/pybind11.h>
