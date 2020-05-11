// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "microsoft.ai.machinelearning.h"
#include "microsoft.ai.machinelearning.native.h"

#include "raw/microsoft.ai.machinelearning.h"
#include "raw/microsoft.ai.machinelearning.gpu.h"

void RunOnDevice(Microsoft::AI::MachineLearning::learning_model& model, Microsoft::AI::MachineLearning::learning_model_device& device, bool copy_inputs);