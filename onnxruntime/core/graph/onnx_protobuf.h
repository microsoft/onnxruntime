// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
// include\google/protobuf/port.h(96,68): warning C4100: 'context': unreferenced formal parameter
#pragma warning(disable : 4100)
#pragma warning(disable : 4146)
#pragma warning(disable : 4127)
#pragma warning(disable : 4267)
#pragma warning(disable : 4244)
#endif

#if !defined(ORT_MINIMAL_BUILD)
#include <onnx/defs/schema.h>
#else
#include <onnx/defs/data_type_utils.h>
#endif

#include <onnx/onnx_pb.h>
#include <onnx/onnx-operators_pb.h>

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif


