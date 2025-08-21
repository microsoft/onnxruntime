// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include <iostream>
#include <sstream>

#include "strings_helper.h"
#include "core/common/common.h"
#include "core/common/parse_string.h"
#include "core/common/string_utils.h"

namespace onnxruntime {
namespace perftest {

void ParseSessionConfigs(const std::string& configs_string,
                         std::unordered_map<std::string, std::string>& session_configs,
                         const std::unordered_set<std::string>& available_keys) {
  std::istringstream ss(configs_string);
  std::string token;

  while (ss >> token) {
    if (token == "") {
      continue;
    }

    std::string_view token_sv(token);

    auto pos = token_sv.find("|");
    if (pos == std::string_view::npos || pos == 0 || pos == token_sv.length()) {
      ORT_THROW("Use a '|' to separate the key and value for the run-time option you are trying to use.\n");
    }

    std::string key(token_sv.substr(0, pos));
    std::string value(token_sv.substr(pos + 1));

    if (available_keys.empty() == false && available_keys.count(key) == 0) {
      // Error: unknown option: {key}
      std::string available_keys_str;
      for (const auto& av_key : available_keys) {
        available_keys_str += av_key;
        available_keys_str += ", ";
      }
      ORT_THROW("[ERROR] wrong key type entered : `", key,
                "`. The following runtime key options are available: [", available_keys_str, "]");
    }

    auto it = session_configs.find(key);
    if (it != session_configs.end()) {
      // Error: specified duplicate session configuration entry: {key}
      ORT_THROW("Specified duplicate session configuration entry: ", key);
    }

    session_configs.insert(std::make_pair(std::move(key), std::move(value)));
  }
}

bool ParseDimensionOverride(const std::string& input, std::map<std::string, int64_t>& free_dim_override_map) {
  std::stringstream ss(input);
  std::string free_dim_str;

  while (std::getline(ss, free_dim_str, ' ')) {
    if (!free_dim_str.empty()) {
      size_t delimiter_location = free_dim_str.find(":");
      if (delimiter_location >= free_dim_str.size() - 1) {
        return false;
      }
      std::string dim_identifier = free_dim_str.substr(0, delimiter_location);
      std::string override_val_str = free_dim_str.substr(delimiter_location + 1, std::string::npos);
      ORT_TRY {
        int64_t override_val = std::stoll(override_val_str.c_str());
        if (override_val <= 0) {
          return false;
        }
        free_dim_override_map[dim_identifier] = override_val;
      }
      ORT_CATCH(const std::exception& ex) {
        ORT_HANDLE_EXCEPTION([&]() {
          std::cerr << "Error parsing free dimension override value: " << override_val_str.c_str() << ", " << ex.what() << std::endl;
        });
        return false;
      }
    }
  }

  return true;
}

bool ParseDimensionOverrideFromArgv(int argc, std::vector<std::string>& argv, std::string& option, std::map<std::string, int64_t>& free_dim_override_map) {
  for (int i = 1; i < argc; ++i) {
    auto utf8_arg = argv[i];
    if (utf8_arg == ("-" + option) || utf8_arg == ("--" + option)) {
      auto value_idx = i + 1;
      if (value_idx >= argc || argv[value_idx][0] == '-') {
        std::cerr << utf8_arg << " should be followed by a key-value pair." << std::endl;
        return false;
      }

      if (!ParseDimensionOverride(argv[value_idx], free_dim_override_map)) return false;
    }
  }
  return true;
}

void ParseEpOptions(const std::string& input, std::vector<std::unordered_map<std::string, std::string>>& result) {
  auto tokens = utils::SplitString(input, ";", true);

  for (const auto& token : tokens) {
    result.emplace_back();  // Adds a new empty map
    if (!token.empty()) {
      ParseSessionConfigs(std::string(token), result.back());  // only parse non-empty
    }
    // if token is empty, we still get an empty map in `result`
  }
}

void ParseEpList(const std::string& input, std::vector<std::string>& result) {
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, ';')) {
    if (!token.empty()) {
      result.push_back(token);
    }
  }
}

void ParseEpDeviceIndexList(const std::string& input, std::vector<int>& result) {
  std::stringstream ss(input);
  std::string item;

  while (std::getline(ss, item, ';')) {
    if (!item.empty()) {
      int value = ParseStringWithClassicLocale<int>(item);
      result.push_back(value);
    }
  }
}
}  // namespace perftest
}  // namespace onnxruntime
