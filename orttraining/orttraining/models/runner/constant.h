// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace training {

constexpr const char* LRSchedule_Cosine = "Cosine";
constexpr const char* LRSchedule_Constant = "Constant";
constexpr const char* LRSchedule_Linear = "Linear";
constexpr const char* LRSchedule_Poly = "Poly";
constexpr const char* LRSchedule_NoWarmup = "None";

}  // namespace training
}  // namespace onnxruntime
