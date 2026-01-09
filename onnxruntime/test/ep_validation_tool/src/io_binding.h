#pragma once

#include "onnxruntime_cxx_api.h"
#include <unordered_map>
#include <string>

std::unordered_map<std::string, Ort::Value> CreateNamedInputs(const Ort::Session& session, OrtAllocator* allocator);
std::unordered_map<std::string, Ort::Value> CreateNamedOutputs(const Ort::Session& session, OrtAllocator* allocator);
std::vector<std::string> CreateTensorNames(const Ort::Session& session, OrtAllocator* allocator, bool is_input);
std::vector<Ort::Value> CreateTensors(const Ort::Session& session, OrtAllocator* allocator, bool is_input);
Ort::IoBinding CreateBinding(
    Ort::Session& session,
    std::unordered_map<std::string, Ort::Value>& inputs,
    std::unordered_map<std::string, Ort::Value>& outputs);
