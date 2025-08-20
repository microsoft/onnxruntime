// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.
#pragma once
#include "core/session/onnxruntime_cxx_api.h"
#include <string_view>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace onnxruntime {
namespace test {
namespace utils {

bool ParseDimensionOverride(const std::string& input, std::map<std::string, int64_t>& free_dim_override_map);

bool ParseDimensionOverrideFromArgv(int argc, std::vector<std::string>& argv, std::string& option, std::map<std::string, int64_t>& free_dim_override_map);

void ParseSessionConfigs(const std::string& configs_string,
                         std::unordered_map<std::string, std::string>& session_configs,
                         const std::unordered_set<std::string>& available_keys = {});

bool ParseDimensionOverride(const std::string& input, std::map<std::string, int64_t>& free_dim_override_map);

bool ParseDimensionOverrideFromArgv(int argc, std::vector<std::string>& argv, std::string& option, std::map<std::string, int64_t>& free_dim_override_map);

void ParseEpList(const std::string& input, std::vector<std::string>& result);

void ParseEpOptions(const std::string& input, std::vector<std::unordered_map<std::string, std::string>>& result);

void ParseEpDeviceIndexList(const std::string& input, std::vector<int>& result);

std::vector<std::string> ConvertArgvToUtf8Strings(int argc, ORTCHAR_T* argv[]);

std::vector<char*> CStringsFromStrings(std::vector<std::string>& utf8_args);

}  // namespace utils
}  // namespace test
}  // namespace onnxruntime
