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

/**
 * @brief Computes the Levenshtein distance between two strings.
 *
 * The Levenshtein distance is a metric for measuring the difference between two strings.
 * It represents the minimum number of single-character edits required to change one string
 * into the other. The possible edits are insertion, deletion, or substitution of a character.
 *
 * This function uses a dynamic programming approach with optimized space complexity.
 * It maintains only two vectors representing the previous and current rows of the distance matrix.
 *
 * @param str1 The first input string.
 * @param str2 The second input string.
 * @return The Levenshtein distance between the two strings.
 */
static size_t LevenShtein(const std::string& str1, const std::string& str2) {
  size_t m = str1.length();
  size_t n = str2.length();

  std::vector<size_t> prevRow(n + 1, 0);
  std::vector<size_t> currRow(n + 1, 0);

  for (size_t j = 0; j <= n; j++) {
    prevRow[j] = j;
  }

  for (size_t i = 1; i <= m; i++) {
    currRow[0] = i;
    for (size_t j = 1; j <= n; j++) {
      if (str1[i - 1] == str2[j - 1]) {
        currRow[j] = prevRow[j - 1];
      } else {
        currRow[j] = 1 + std::min(currRow[j - 1], std::min(prevRow[j], prevRow[j - 1]));
      }
    }
    prevRow = currRow;
  }

  return currRow[n];
}

static std::string FindNearestKey(const std::string& key, const std::unordered_set<std::string>& available_keys) {
  std::string similar_key;
  size_t min_distance = std::numeric_limits<int>::max();
  for (const auto& av_key : available_keys) {
    size_t distance = LevenShtein(av_key, key);
    if (distance < min_distance) {
      min_distance = distance;
      similar_key = av_key;
    }
  }
  return similar_key;
}

bool ParseSessionConfigs(const std::string& configs_string,
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
      ORT_THROW("[ERROR] wrong key type entered : `", key, "`. Did you mean: `", FindNearestKey(key, available_keys),
                "`?. The following runtime key options are avaible: [", available_keys_str, "]");
    }

    auto it = session_configs.find(key);
    if (it != session_configs.end()) {
      // Error: specified duplicate session configuration entry: {key}
      ORT_THROW("Specified duplicate session configuration entry: ", key);
    }

    session_configs.insert(std::make_pair(std::move(key), std::move(value)));
  }

  return true;
}
}  // namespace perftest
}  // namespace onnxruntime
