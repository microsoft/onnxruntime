#pragma once

#include "onnxruntime_cxx_api.h"

#include <filesystem>
#include <unordered_map>

using NamedTensors = std::unordered_map<std::string, Ort::Value>;

Ort::Value ReadNumpy(const std::filesystem::path& path);

bool SaveOrtValueAsNumpyArray(const std::filesystem::path& path, const Ort::Value& value);

void CastOrtValueData(const Ort::Value& value, std::vector<float>& buffer);

void CopyTensorMap(
    const std::unordered_map<std::string, Ort::Value>& source, std::unordered_map<std::string, Ort::Value>& dest);

Ort::Value CopyTensor(const Ort::Value& source);

bool AllNans(const Ort::Value& tensor);
bool AnyNans(const Ort::Value& tensor);
