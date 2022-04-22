// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// If performing a debug build with VS2022 (_MSC_VER == 1930) we need to include corecrt.h before pybind
// so that the _STL_ASSERT macro is defined in a compatible way.
//
// pybind11/pybind11.h includes pybind11/detail/common.h, which undefines _DEBUG whilst including the Python headers
// (which in turn include corecrt.h). This alters how the _STL_ASSERT macro is defined and causes the build to fail.
//
// see https://github.com/microsoft/onnxruntime/issues/9735
//
#if defined(_MSC_VER) && defined(_DEBUG) && _MSC_VER >= 1930
#include <corecrt.h>
#endif

#include <pybind11/pybind11.h>
