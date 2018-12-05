// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"

namespace onnxruntime {
ONNXStatus* ToONNXStatus(const onnxruntime::common::Status& st);
};
