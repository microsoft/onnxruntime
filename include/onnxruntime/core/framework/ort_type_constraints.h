// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/onnxruntime_c_api.h"
#include <unordered_map>
#include <string>
#include <set>

struct OrtTypeConstraints {
    bool AddTypeConstraint(const char* type_symbol, ONNXTensorElementDataType type);
    inline const std::unordered_map<std::string, std::set<ONNXTensorElementDataType>>& GetTypeConstraints() const { return type_constraints_; };
private:
    std::unordered_map<std::string, std::set<ONNXTensorElementDataType>> type_constraints_;
};
