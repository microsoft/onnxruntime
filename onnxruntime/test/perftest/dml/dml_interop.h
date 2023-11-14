// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/session/onnxruntime_cxx_api.h>

std::pair<Ort::Value, std::unique_ptr<void, void (*)(void*)>> CreateDmlValue(
    const Ort::ConstTensorTypeAndShapeInfo& tensor_info,
    const Ort::Session& session,
    Ort::Value&& default_value,
    const char* output_name,
    bool is_input);

std::pair<Ort::Value, std::unique_ptr<void, void (*)(void*)>> CreateDmlValueFromCpuValue(
    Ort::Value&& cpu_value,
    const Ort::Session& session,
    const char* input_name);
