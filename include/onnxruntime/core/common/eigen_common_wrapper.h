//-----------------------------------------------------------------------------
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
//-----------------------------------------------------------------------------
#pragma once
#include "onnxruntime_config.h"
// build/external/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:162:71:
// error: ignoring attributes on template argument "Eigen::PacketType<const float, Eigen::DefaultDevice>::type {aka __vector(4) float}" [-Werror=ignored-attributes]
#if defined(__GNUC__)
#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"
#ifdef HAS_DEPRECATED_COPY
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#endif
#elif defined(_MSC_VER)
// build\windows\debug\external\eigen3\unsupported\eigen\cxx11\src/Tensor/Tensor.h(76):
// warning C4554: '&': check operator precedence for possible error; use parentheses to clarify precedence

// unsupported\eigen\cxx11\src\Tensor\TensorUInt128.h(150,0): Warning C4245: 'initializing': conversion from '__int64'
// to 'uint64_t', signed/unsigned mismatch
#pragma warning(push)
#pragma warning(disable : 4554)
#pragma warning(disable : 4245)
#pragma warning(disable : 4127)
#pragma warning(disable : 4805)
#pragma warning(disable : 6313)
#pragma warning(disable : 6294)
#endif

#include "unsupported/Eigen/CXX11/Tensor"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
