// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <functional>
#include "gradient_builder_base.h"

namespace onnxruntime {
namespace training {

typedef std::function<std::unique_ptr<GradientBuilderBase>(const Node*,
                                                           const std::unordered_set<std::string>&,
                                                           const std::unordered_set<std::string>&)>
    GradientBuilderFn;

class GradientBuilderRegistry {
 public:
  static GradientBuilderRegistry& GetGradientBuilderRegistry() {
    static GradientBuilderRegistry gradient_builder_registry;
    return gradient_builder_registry;
  }

  GradientBuilderRegistry() = default;

  void RegisterGradientBuilderFunc(const std::string& op, GradientBuilderFn fn) {
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
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GradientBuilderRegistry);
  std::unordered_map<std::string, GradientBuilderFn> gradient_builder_map;
};

// TODO: check better way to "new" a gradientbuilder here
// raw pointer not good
#define REGISTER_GRADIENT_BUILDER(op, gradientbuilder)                                     \
  GradientBuilderRegistry::GetGradientBuilderRegistry().RegisterGradientBuilderFunc(       \
      op,                                                                                  \
      [](const Node* node,                                                                 \
         const std::unordered_set<std::string>& gradient_inputs,                           \
         const std::unordered_set<std::string>& gradient_outputs)                          \
          -> std::unique_ptr<GradientBuilderBase> {                                        \
        return std::make_unique<gradientbuilder>(node, gradient_inputs, gradient_outputs); \
      });

#define NO_GRADIENT(op) REGISTER_GRADIENT_BUILDER(op, EmptyGradientBuilder)

// There are some operators which are not really computation operators and one shouldn't attempt to
// request one for such operators.
#define SHOULD_NOT_DO_GRADIENT(op) REGISTER_GRADIENT_BUILDER(op, UnSupportedGradientBuilder)

GradientDef GetGradientForOp(const Node* node,
                             const std::unordered_set<std::string>& output_args_need_grad,
                             const std::unordered_set<std::string>& input_args_need_grad);

void RegisterGradientBuilders();
}  // namespace training
}  // namespace onnxruntime
