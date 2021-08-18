// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

namespace onnxruntime {
namespace training {

// Since the final attribute value and dtype is determined at runtime for some cases, e.g., when the dtype is
// IElemType(i) or OElemType(i), here we keep the value and dtype in string format and will parse it during runtime.
// Value is a JSON string for easy parsing.
struct GradientNodeAttributeDefinition {
  std::string name;
  std::string value_json;
  std::string dtype;
  bool is_tensor;
};

struct GradientNodeDefinition {
  std::string op_type;
  std::string domain;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<GradientNodeAttributeDefinition> attributes;
};

class GradientDefinitionRegistry {
 public:
  static GradientDefinitionRegistry& Instance() {
    static GradientDefinitionRegistry instance;
    return instance;
  }

  const std::vector<GradientNodeDefinition>* GetGradientDefinition(const std::string& key) {
    auto it = definitions_.find(key);
    if (it == definitions_.end()) {
      return nullptr;
    }

    return &it->second;
  }

  bool Contains(const std::string& key) {
    return definitions_.find(key) != definitions_.end();
  }

  void Register(const std::string& key, const std::vector<GradientNodeDefinition>& definition) {
    // It's possible the new definition overwrite the old one.
    definitions_.emplace(key, definition);
  }

 private:
  std::unordered_map<std::string, std::vector<GradientNodeDefinition>> definitions_;
};

}  // namespace training
}  // namespace onnxruntime
