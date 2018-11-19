// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/error_code.h"

namespace onnxruntime {
ONNXStatusPtr ToONNXStatus(const onnxruntime::common::Status& st);
};
