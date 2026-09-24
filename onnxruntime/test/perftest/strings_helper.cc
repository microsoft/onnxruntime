// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include <charconv>
#include <iostream>
#include <sstream>
#include <system_error>

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

void ParseEpDeviceFilterKeyValuePairs(const std::string& input, std::vector<std::pair<std::string, std::string>>& result) {
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, ' ')) {
    if (!token.empty()) {
      size_t delimiter_location = token.find("|");
      if (delimiter_location == std::string::npos || delimiter_location == 0 || delimiter_location == token.size() - 1) {
        ORT_THROW("Use a '|' to separate the key and value for the device filter you are trying to use.\n");
      }
      std::string key = token.substr(0, delimiter_location);
      std::string value = token.substr(delimiter_location + 1);
      result.emplace_back(std::make_pair(std::move(key), std::move(value)));
    }
  }
}

bool ParseDataShapeGroups(const std::string& input,
                          std::map<std::string, std::vector<std::vector<int64_t>>>& data_shape_groups) {
  // Parse format: "input_name:[d0,d1,...][d0,d1,...] input_name2:[d0,d1][d0,d1]"
  // Split on whitespace that is outside brackets to allow spaces inside shape specs.
  std::vector<std::string> input_specs;
  std::string current;
  int bracket_depth = 0;
  for (char c : input) {
    if (c == '[') {
      bracket_depth++;
      current += c;
    } else if (c == ']') {
      bracket_depth--;
      current += c;
    } else if ((c == ' ' || c == '\t') && bracket_depth == 0) {
      if (!current.empty()) {
        input_specs.push_back(current);
        current.clear();
      }
    } else {
      current += c;
    }
  }
  if (!current.empty()) {
    input_specs.push_back(current);
  }

  if (bracket_depth != 0) {
    std::cerr << "Error parsing --data_shape: mismatched brackets (unbalanced '[' and ']')." << std::endl;
    return false;
  }

  for (const auto& input_spec : input_specs) {
    // Split on first ':' to get input_name and shapes string
    size_t colon_pos = input_spec.find(':');
    if (colon_pos == std::string::npos || colon_pos == 0) {
      std::cerr << "Error parsing --data_shape: expected 'name:[shape]...' format, got '" << input_spec << "'." << std::endl;
      return false;
    }

    std::string input_name = input_spec.substr(0, colon_pos);
    std::string shapes_str = input_spec.substr(colon_pos + 1);

    if (shapes_str.empty()) {
      std::cerr << "Error parsing --data_shape: no shape groups provided for input '" << input_name << "'." << std::endl;
      return false;
    }

    std::vector<std::vector<int64_t>> shape_groups;

    // Extract bracket-delimited shape groups: [d0,d1,...][d0,d1,...]
    size_t pos = 0;
    while (pos < shapes_str.size()) {
      while (pos < shapes_str.size() && (shapes_str[pos] == ' ' || shapes_str[pos] == '\t')) {
        ++pos;
      }
      if (pos >= shapes_str.size()) break;
      if (shapes_str[pos] != '[') {
        std::cerr << "Error parsing --data_shape: expected '[' at position " << pos
                  << " in shapes for input '" << input_name << "'." << std::endl;
        return false;
      }

      size_t close_bracket = shapes_str.find(']', pos);
      if (close_bracket == std::string::npos) {
        std::cerr << "Error parsing --data_shape: unmatched bracket for input '" << input_name << "'." << std::endl;
        return false;
      }

      std::string dims_str = shapes_str.substr(pos + 1, close_bracket - pos - 1);
      if (dims_str.empty()) {
        std::cerr << "Error parsing --data_shape: empty shape group for input '" << input_name << "'." << std::endl;
        return false;
      }

      // Parse comma-separated dimensions
      std::vector<int64_t> dims;
      std::stringstream dims_ss(dims_str);
      std::string dim_token;
      while (std::getline(dims_ss, dim_token, ',')) {
        // Trim whitespace from dimension token
        size_t start = dim_token.find_first_not_of(" \t");
        size_t end = dim_token.find_last_not_of(" \t");
        if (start == std::string::npos) {
          std::cerr << "Error parsing --data_shape: empty dimension token for input '" << input_name << "'." << std::endl;
          return false;
        }
        dim_token = dim_token.substr(start, end - start + 1);

        int64_t dim_val = 0;
        const char* begin = dim_token.data();
        const char* end_ptr = begin + dim_token.size();
        auto [ptr, ec] = std::from_chars(begin, end_ptr, dim_val);
        if (ec != std::errc{} || ptr != end_ptr) {
          std::cerr << "Error parsing --data_shape: invalid dimension value '"
                    << dim_token << "' for input '" << input_name << "'." << std::endl;
          return false;
        }
        if (dim_val <= 0) {
          std::cerr << "Error parsing --data_shape: dimensions must be positive integers, got '"
                    << dim_token << "' for input '" << input_name << "'." << std::endl;
          return false;
        }
        dims.push_back(dim_val);
      }

      shape_groups.push_back(std::move(dims));
      pos = close_bracket + 1;
    }

    if (shape_groups.empty()) {
      std::cerr << "Error parsing --data_shape: no shape groups found for input '" << input_name << "'." << std::endl;
      return false;
    }

    if (data_shape_groups.count(input_name) > 0) {
      std::cerr << "Error parsing --data_shape: duplicate input name '" << input_name << "'." << std::endl;
      return false;
    }

    data_shape_groups[input_name] = std::move(shape_groups);
  }

  if (data_shape_groups.empty()) {
    std::cerr << "Error parsing --data_shape: no input specifications found." << std::endl;
    return false;
  }

  // Validate all inputs have the same number of shape groups
  if (data_shape_groups.size() > 1) {
    size_t expected_count = data_shape_groups.begin()->second.size();
    for (const auto& [name, groups] : data_shape_groups) {
      if (groups.size() != expected_count) {
        std::cerr << "Error parsing --data_shape: all inputs must have the same number of shape groups. "
                  << "Input '" << data_shape_groups.begin()->first << "' has " << expected_count
                  << " groups but input '" << name << "' has " << groups.size() << "." << std::endl;
        return false;
      }
    }
  }

  return true;
}

std::string FormatShapeGroup(const std::map<std::string, std::vector<std::vector<int64_t>>>& groups, size_t g) {
  if (groups.empty() || g >= groups.begin()->second.size()) {
    return "";
  }
  std::string result;
  for (const auto& [name, shapes] : groups) {
    if (!result.empty()) result += ", ";
    result += name + ":[";
    const auto& dims = shapes[g];
    for (size_t d = 0; d < dims.size(); d++) {
      if (d > 0) result += ",";
      result += std::to_string(dims[d]);
    }
    result += "]";
  }
  return result;
}

}  // namespace perftest
}  // namespace onnxruntime
