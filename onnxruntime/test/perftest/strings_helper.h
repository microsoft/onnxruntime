// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace onnxruntime {
namespace perftest {

void ParseSessionConfigs(const std::string& configs_string,
                         std::unordered_map<std::string, std::string>& session_configs,
                         const std::unordered_set<std::string>& available_keys = {});

bool ParseDimensionOverride(const std::string& input, std::map<std::string, int64_t>& free_dim_override_map);

bool ParseDimensionOverrideFromArgv(int argc, std::vector<std::string>& argv, std::string& option, std::map<std::string, int64_t>& free_dim_override_map);

void ParseEpList(const std::string& input, std::vector<std::string>& result);

void ParseEpOptions(const std::string& input, std::vector<std::unordered_map<std::string, std::string>>& result);

void ParseEpDeviceIndexList(const std::string& input, std::vector<int>& result);

void ParseEpDeviceFilterKeyValuePairs(const std::string& input, std::vector<std::pair<std::string, std::string>>& result);

bool ParseDataShapeGroups(const std::string& input,
                          std::map<std::string, std::vector<std::vector<int64_t>>>& data_shape_groups);

std::string FormatShapeGroup(const std::map<std::string, std::vector<std::vector<int64_t>>>& groups, size_t g);

}  // namespace perftest
}  // namespace onnxruntime
