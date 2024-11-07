// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include <iostream>
#include <sstream>

#include "strings_helper.h"

namespace onnxruntime {
namespace perftest {

bool ParseSessionConfigs(const std::string& configs_string,
                         std::unordered_map<std::string, std::string>& session_configs) {
  std::istringstream ss(configs_string);
  std::string token;

  while (ss >> token) {
    if (token == "") {
      continue;
    }

    std::string_view token_sv(token);

    auto pos = token_sv.find("|");
    if (pos == std::string_view::npos || pos == 0 || pos == token_sv.length()) {
      // Error: must use a '|' to separate the key and value for session configuration entries.
      return false;
    }

    std::string key(token_sv.substr(0, pos));
    std::string value(token_sv.substr(pos + 1));

    auto it = session_configs.find(key);
    if (it != session_configs.end()) {
      // Error: specified duplicate session configuration entry: {key}
      return false;
    }

    session_configs.insert(std::make_pair(std::move(key), std::move(value)));
  }

  return true;
}
}  // namespace perftest
}  // namespace onnxruntime
