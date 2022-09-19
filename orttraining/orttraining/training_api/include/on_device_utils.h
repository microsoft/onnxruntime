// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/path_lib.h"
#include "core/platform/env.h"
#include "orttraining/training_api/include/module.h"

namespace onnxruntime {
namespace training {
namespace api {

/**
 * @brief Calculates the difference between 2 parameter buffers.
 * @param output_params Parameters buffer for the new model.
 * @param old_output_params Parameters buffer for the old model.
 * @param output Parameters buffer object that will hold the delta values.
 * @return Status
 */

Status GetParametersDifference(const OrtValue output_params, const OrtValue old_output_params, OrtValue& output);

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
