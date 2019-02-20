// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace GradientOps {

typedef std::pair<std::string, int> DefsMapping;

class GradientOpSchema {
 public:
  GradientOpSchema(){};
  GradientOpSchema(const std::vector<DefsMapping>& input_mappings,
                   const std::vector<DefsMapping>& output_mappings)
      : input_mappings_(input_mappings),
        output_mappings_(output_mappings) {}

  std::vector<DefsMapping> InputMappings() { return input_mappings_; }
  std::vector<DefsMapping> OutputMappings() { return output_mappings_; }

 private:
  std::vector<DefsMapping> input_mappings_;
  std::vector<DefsMapping> output_mappings_;
};

struct GradOpSchemaRegistryHelper {
  static std::pair<std::string, int> I(int index) { return std::make_pair("I", index); }
  static std::pair<std::string, int> GI(int index) { return std::make_pair("GI", index); }
  static std::pair<std::string, int> GO(int index) { return std::make_pair("GO", index); }
  static std::pair<std::string, int> O(int index) { return std::make_pair("O", index); }

  static std::unordered_map<std::string, GradientOpSchema> GradientOpRegistry;
};

void RegisterGradientSchemas();

}  // namespace GradientOps
}  // namespace onnxruntime
