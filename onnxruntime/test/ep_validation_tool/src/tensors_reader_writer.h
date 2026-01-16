#pragma once

#include "onnxruntime_cxx_api.h"

#include <string>
#include <unordered_map>

class ITensorsWriter
{
public:
    virtual bool Store(size_t sample_idx, const std::string& tensor_name, const Ort::Value& tensors) const = 0;
};

class ITensorsReader
{
public:
    virtual bool Load(size_t sample_idx, const std::string& tensor_name, Ort::Value& tensor) const = 0;
    virtual size_t GetNumSamples() const = 0;
};
