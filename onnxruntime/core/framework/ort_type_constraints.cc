// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/ort_type_constraints.h"

bool OrtTypeConstraints::AddTypeConstraint(const char* type_symbol, ONNXTensorElementDataType type) {
    std::unordered_map<std::string, std::set<ONNXTensorElementDataType>>::iterator iter = type_constraints_.find(type_symbol);
    if (iter == type_constraints_.end()) {
        std::set<ONNXTensorElementDataType> types{type};
        type_constraints_[type_symbol] = types;
        return true;
    }
    return (iter->second).insert(type).second;
}
