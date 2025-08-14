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
namespace test {
namespace utils {

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

void ParseEpOptions(const std::string& input, std::vector<std::unordered_map<std::string, std::string>>& result) {
  auto tokens = onnxruntime::utils::SplitString(input, ";", true);

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

std::vector<std::string> ConvertArgvToUtf8Strings(int argc, ORTCHAR_T* argv[]) {
  std::vector<std::string> utf8_args;
  utf8_args.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    std::string utf8_string = ToUTF8String(argv[i]);

    // Abseil flags doens't natively alias "-h" to "--help".
    // We make "-h" alias to "--help" here.
    if (utf8_string == "-h" || utf8_string == "--h") {
      utf8_args.push_back("--help");
    } else {
      utf8_args.push_back(utf8_string);
    }
  }
  return utf8_args;
}

std::vector<char*> CStringsFromStrings(std::vector<std::string>& utf8_args) {
  std::vector<char*> utf8_argv;
  utf8_argv.reserve(utf8_args.size());
  for (auto& str : utf8_args) {
    utf8_argv.push_back(&str[0]);
  }
  return utf8_argv;
}

}  // namespace utils
}  // namespace test
}  // namespace onnxruntime
