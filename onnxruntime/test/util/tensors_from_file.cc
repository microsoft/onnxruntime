// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include "test/util/include/tensors_from_file.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace test {

static inline void TrimLeft(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
}

static inline void TrimRight(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
          }).base(),
          s.end());
}

static inline void Trim(std::string& s) {
  TrimRight(s);
  TrimLeft(s);
}

static inline bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.compare(0, prefix.size(), prefix) == 0;
}

void LoadTensorsFromFile(const std::string& path, std::unordered_map<std::string, std::vector<float>>& tensors) {
  std::ifstream infile(path);
  if (!infile.good()) {
    ORT_THROW("Cannot open file:", path);
  }

  // File format is like the following:
  //   name:TestCaseName1.TensorName1
  //   1.1, 2.3
  //   ===
  //   name:TestCaseName2.TensorName2
  //   3.3, 4.5,
  //   5.6, 6.7
  //   ===
  // Note that "name:" and "===" shall be at the beginning of a line without leading space!
  const std::string name_prefix = "name:";
  const std::string end_tensor_prefix = "===";

  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }

    std::string name;
    if (StartsWith(line, name_prefix)) {
      name = line.substr(name_prefix.length());
      Trim(name);
    }
    if (name.empty()) {
      ORT_THROW("Failed to find name in line:", line);
    }

    std::vector<float> values;
    while (std::getline(infile, line)) {
      if (line.empty()) {
        continue;
      }
      if (StartsWith(line, end_tensor_prefix)) {
        break;
      }

      std::istringstream ss(line);
      for (std::string token; std::getline(ss, token, ',');) {
        Trim(token);
        if (token.empty()) {
          continue;
        }

        ORT_TRY {
          float value = std::stof(token);
          values.push_back(value);
        }
        ORT_CATCH(const std::exception&) {
          ORT_THROW("Failed to parse float from name='", name, ", token='", token);
        }
      }
    }

    if (values.size() == 0) {
      ORT_THROW("No values for name=", name);
    }

    auto result = tensors.insert({name, values});
    if (result.second == false) {
      ORT_THROW("Failed to insert: duplicated name=", name);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
