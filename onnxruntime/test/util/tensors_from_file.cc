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

static inline void ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
}

static inline void rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
          }).base(),
          s.end());
}

static inline void trim(std::string& s) {
  rtrim(s);
  ltrim(s);
}

static inline bool starts_with(const std::string& s, const std::string& prefix) {
  return s.compare(0, prefix.size(), prefix) == 0;
};

// File format is like the following:
//   name:TestCaseName1.TensorName1
//   1.1, 2.3
//   ===
//   name:TestCaseName2.TensorName2
//   3.3, 4.5,
//   5.6, 6.7
//   ===
void load_tensors_from_file(const std::string& path, std::unordered_map<std::string, std::vector<float>>& tensors) {
  std::ifstream infile(path);
  if (!infile.good()) {
    ORT_THROW("Cannot open file:", path);
  }

  const std::string prefix = "name:";
  const std::string end_tensor_prefix = "===";

  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }

    std::string name;
    if (starts_with(line, prefix)) {
      name = line.substr(prefix.length());
      trim(name);
    }
    if (name.empty()) {
      ORT_THROW("Empty name in line:", line);
    }

    std::vector<float> values;
    while (std::getline(infile, line)) {
      if (line.empty()) {
        continue;
      }
      if (starts_with(line, end_tensor_prefix)) {
        break;
      }

      std::istringstream ss(line);
      for (std::string token; std::getline(ss, token, ',');) {
        trim(token);
        if (token.empty()) {
          continue;
        }

        try {
          float value = std::stof(token);
          values.push_back(value);
        } catch (const std::exception& e) {
          ORT_THROW("Failed to parse float from name='", name, ",token='", token, "'. Exception:", e.what());
        }
      }
    }

    if (values.size() == 0) {
      ORT_THROW("No values for name=", name);
    }

    auto result = tensors.insert({name, values});
    if (result.second == false) {  // not inserted
      ORT_THROW("Failed to insert name=", name);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
