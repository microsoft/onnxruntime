// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "std.h"

namespace ProtobufHelpers
{
    // LoadTensorFromProtobufFile take a path to a FP32 data file and loads it into a 32bit array or
    // 16bit array based on isFp16
    winrt::Windows::AI::MachineLearning::ITensor LoadTensorFromProtobufFile(const std::wstring& filePath, bool isFp16);
    // LoadTensorFloat16FromProtobufFile takes a path to a FP16 data file and loads it into a 16bit array
    winrt::Windows::AI::MachineLearning::TensorFloat16Bit LoadTensorFloat16FromProtobufFile(const std::wstring& filePath);

    winrt::Windows::AI::MachineLearning::LearningModel CreateModel(
        winrt::Windows::AI::MachineLearning::TensorKind kind,
        const std::vector<int64_t>& shape,
        uint32_t num_elements = 1);
}
