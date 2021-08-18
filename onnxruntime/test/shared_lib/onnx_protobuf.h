// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
// TODO(): delete this file from public interface
#ifdef __GNUC__
#include "onnxruntime_config.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#ifdef HAS_DEPRECATED_DECLARATIONS
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#endif
#include "onnx/onnx-ml.pb.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
