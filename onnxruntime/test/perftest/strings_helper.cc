// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include <iostream>
#include <sstream>

#include "strings_helper.h"
#include "core/common/common.h"

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
                "`. The following runtime key options are avaible: [", available_keys_str, "]");
    }

    auto it = session_configs.find(key);
    if (it != session_configs.end()) {
      // Error: specified duplicate session configuration entry: {key}
      ORT_THROW("Specified duplicate session configuration entry: ", key);
    }

    session_configs.insert(std::make_pair(std::move(key), std::move(value)));
  }
}
}  // namespace perftest
}  // namespace onnxruntime
