// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
// include the pybind header first, it will disable linking to pythonX_d.lib on Windows in debug mode
#include "python/onnxruntime_pybind_state_common.h"
#include <torch/extension.h>
#include <ATen/Operators.h>