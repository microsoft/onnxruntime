//-----------------------------------------------------------------------------
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
//-----------------------------------------------------------------------------
#pragma once

// build/external/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:162:71:
// error: ignoring attributes on template argument "Eigen::PacketType<const float, Eigen::DefaultDevice>::type {aka __vector(4) float}" [-Werror=ignored-attributes]
#if defined(__GNUC__)
#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
// build\windows\debug\external\eigen3\unsupported\eigen\cxx11\src/Tensor/Tensor.h(76):
// warning C4554: '&': check operator precedence for possible error; use parentheses to clarify precedence
// build\windows\debug\external\eigen3\unsupported\eigen\cxx11\src/Tensor/TensorStorage.h(65):
// warning C4324: structure was padded due to alignment specifier
// unsupported\eigen\cxx11\src\Tensor\TensorUInt128.h(150,0): Warning C4245: 'initializing': conversion from '__int64'
// to 'uint64_t', signed/unsigned mismatch
#pragma warning(push)
#pragma warning(disable : 4554)
#pragma warning(disable : 4324)
#pragma warning(disable : 4245)
#endif

#include "unsupported/Eigen/CXX11/Tensor"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
