// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_config.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push

#ifdef HAS_SHORTEN_64_TO_32
#pragma GCC diagnostic ignored "-Wshorten-64-to-32"
#endif

#endif

#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/schema.h"
#else
#include "onnx/defs/data_type_utils.h"
#endif

#include "onnx/onnx_pb.h"
#include "onnx/onnx-operators_pb.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
