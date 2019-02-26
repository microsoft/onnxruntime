// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>
#include <functional>
#include "gradient_builder_base.h"

namespace onnxruntime {

typedef std::function<GradientBuilderBase*(const Node*, const std::unordered_set<std::string>&, const std::unordered_set<std::string>&)> GradientBuilderFn;

class GradientBuilderRegistry {
 public:
  static GradientBuilderRegistry& GetGradientBuilderRegistry() {
    static GradientBuilderRegistry gradient_builder_registry;
    return gradient_builder_registry;
  }

  void
  RegisterGradientBuilderFunc(const std::string& op, GradientBuilderFn fn) {
    gradient_builder_map.insert(std::make_pair(op, fn));
  }

  GradientBuilderFn GetGradientBuilderFunc(const std::string& op) {
    auto creator = gradient_builder_map.find(op);
    if (creator != gradient_builder_map.end()) {
      return creator->second;
    } else {
      ORT_THROW("op does not have a registered gradient");
    }
  }

 private:
  std::unordered_map<std::string, GradientBuilderFn> gradient_builder_map;
};

#define REGISTER_GRADIENT_BUILDER(op, gradientbuilder)                               \
  GradientBuilderRegistry::GetGradientBuilderRegistry().RegisterGradientBuilderFunc( \
      op,                                                                            \
      [](const Node* node,                                                           \
         const std::unordered_set<std::string>& gradient_inputs,                     \
         const std::unordered_set<std::string>& gradient_outputs)                    \
          -> GradientBuilderBase* {                                                  \
        return new gradientbuilder(node, gradient_inputs, gradient_outputs);         \
      });

#define NO_GRADIENT(op) REGISTER_GRADIENT_BUILDER(op, EmptyGradientBuilder)
}  // namespace onnxruntime
