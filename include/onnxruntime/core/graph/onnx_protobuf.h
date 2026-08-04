// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_config.h"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push

#ifdef HAS_SHORTEN_64_TO_32
#pragma GCC diagnostic ignored "-Wshorten-64-to-32"
#endif

#endif

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_USE_ONNX_LIGHT)
#include "onnx/defs/schema.h"
#elif !defined(ORT_USE_ONNX_LIGHT)
#include "onnx/defs/data_type_utils.h"
#endif

#if defined(ORT_USE_ONNX_LIGHT)
// onnx-light does not use protobuf; proto types come from onnx_lib/common/onnx_pb.h
// which includes the hand-crafted message definitions in onnx.h.
#include "onnx_lib/common/onnx_pb.h"
#else
#include "onnx/onnx_pb.h"
#include "onnx/onnx-operators_pb.h"
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifdef _WIN32
#pragma warning(pop)
#endif
